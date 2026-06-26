import node_helpers
import comfy.utils
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import comfy.model_management
import comfy.samplers
import torch
import math
import nodes
import comfy.ldm.flux.math

class CLIPTextEncodeFlux(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLIPTextEncodeFlux",
            category="model/conditioning/flux",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("clip_l", multiline=True, dynamic_prompts=True),
                io.String.Input("t5xxl", multiline=True, dynamic_prompts=True),
                io.Float.Input("guidance", default=3.5, min=0.0, max=100.0, step=0.1),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, clip_l, t5xxl, guidance) -> io.NodeOutput:
        tokens = clip.tokenize(clip_l)
        tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]

        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens, add_dict={"guidance": guidance}))

    encode = execute  # TODO: remove

class EmptyFlux2LatentImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyFlux2LatentImage",
            display_name="Empty Flux 2 Latent",
            category="model/latent/flux",
            inputs=[
                io.Int.Input("width", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, width, height, batch_size=1) -> io.NodeOutput:
        latent = torch.zeros([batch_size, 128, height // 16, width // 16], device=comfy.model_management.intermediate_device())
        return io.NodeOutput({"samples": latent})

class EmptySefiLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptySefiLatent",
            display_name="Empty SeFi Latent",
            category="model/latent/flux",
            description="Empty latent for SeFi-Image (semantic + texture channels, read from the model).",
            inputs=[
                io.Model.Input("model", tooltip="The SeFi-Image diffusion model (sets the channel count)."),
                io.Int.Input("width", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, model: io.Model.Type, width, height, batch_size=1) -> io.NodeOutput:
        # SeFi denoises [semantic | texture]; total channels are derived from the checkpoint.
        channels = model.model.model_config.unet_config["in_channels"]
        latent = torch.zeros([batch_size, channels, height // 16, width // 16], device=comfy.model_management.intermediate_device())
        return io.NodeOutput({"samples": latent})


def _sefi_semantic_first_sampler(delta_t, semantic_channels, timestep_shift_alpha):
    """Build a KSAMPLER that runs SeFi's semantic-first dual-rate update.

    The latent is [semantic | texture]. The semantic stream leads the texture
    stream by ``delta_t`` on its own sigma schedule (the reference's dual-rate
    loop), so the texture stream denoises against an already-cleaner semantic
    anchor — which the model was trained for. Each stream is advanced by its own
    d-sigma; the per-stream velocity is recovered from the model's denoised output
    (sigma_tex is used for the whole batch, so v = (x - denoised)/sigma_tex is the
    raw velocity for both halves). The semantic timestep is handed to the DiT via
    transformer_options["sefi_t_sem"]. SeFi trains on a linear flow schedule, so
    the schedule is rebuilt linearly here (the connected scheduler only sets the
    step count and sigma_max); timestep_shift_alpha applies the base/RL warp.
    """
    @torch.no_grad()
    def sample(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
        extra_args = {} if extra_args is None else extra_args
        steps = len(sigmas) - 1
        if steps < 1:
            return x
        sig_max = float(sigmas[0])
        u = torch.linspace(0.0, 1.0, steps + 1, device=x.device, dtype=torch.float64)
        if timestep_shift_alpha != 1.0:                      # base/RL schedule warp
            a = timestep_shift_alpha
            u = (a * u) / (1.0 + (a - 1.0) * u)
        u_sem = torch.clamp(u * (1.0 + delta_t), max=1.0)            # semantic leads
        u_tex = torch.clamp(u * (1.0 + delta_t) - delta_t, min=0.0, max=1.0)
        sig_sem = ((1.0 - u_sem) * sig_max).to(x.dtype)
        sig_tex = ((1.0 - u_tex) * sig_max).to(x.dtype)
        mo = extra_args.setdefault("model_options", {})
        to = mo.setdefault("transformer_options", {})
        s = semantic_channels
        for i in range(steps):
            s_tex, s_tex_n = sig_tex[i], sig_tex[i + 1]
            s_sem, s_sem_n = sig_sem[i], sig_sem[i + 1]
            # tell the DiT the semantic stream's (cleaner) timestep == its sigma (FLUX sampling)
            to["sefi_t_sem"] = s_sem.reshape(1).expand(x.shape[0]).clone()
            sigma_in = s_tex.reshape(1).expand(x.shape[0]).clone()
            denoised = model(x, sigma_in, **extra_args)
            velocity = (x - denoised) / s_tex.clamp(min=1e-6)
            nx = x.clone()
            nx[:, :s] = x[:, :s] + (s_sem_n - s_sem) * velocity[:, :s]   # semantic rate
            nx[:, s:] = x[:, s:] + (s_tex_n - s_tex) * velocity[:, s:]   # texture rate
            x = nx
            if callback is not None:
                callback({"x": x, "i": i, "sigma": s_tex, "sigma_hat": s_tex, "denoised": denoised})
        to.pop("sefi_t_sem", None)
        return x
    return comfy.samplers.KSAMPLER(sample)


class SamplerSefiSemanticFirst(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplerSefiSemanticFirst",
            display_name="SeFi Semantic-First Sampler",
            category="sampling/custom_sampling/samplers",
            description="SeFi-Image semantic-first dual-rate sampler (use with SamplerCustomAdvanced). "
                        "Runs the model in-distribution (semantic stream leads texture) for sharper, "
                        "more natural detail than the stock lockstep path.",
            inputs=[
                io.Model.Input("model", tooltip="The SeFi-Image model (provides semantic channels + trained delta_t)."),
                io.Float.Input("delta_t", default=-1.0, min=-1.0, max=1.0, step=0.01,
                               tooltip="Semantic-first lead. -1 = use the checkpoint's trained value."),
                io.Float.Input("timestep_shift_alpha", default=1.0, min=0.01, max=10.0, step=0.01,
                               tooltip="Schedule warp: 1.0 for turbo, 0.3 for base/RL."),
            ],
            outputs=[io.Sampler.Output()],
        )

    @classmethod
    def execute(cls, model: io.Model.Type, delta_t=-1.0, timestep_shift_alpha=1.0) -> io.NodeOutput:
        unet_config = model.model.model_config.unet_config
        semantic_channels = unet_config.get("semantic_channels", 0)
        if semantic_channels <= 0:
            raise ValueError("SamplerSefiSemanticFirst requires a SeFi-Image model (no semantic channels found).")
        if delta_t < 0.0:
            delta_t = float(getattr(model.model.diffusion_model, "sefi_delta_t_trained", 0.1))
        return io.NodeOutput(_sefi_semantic_first_sampler(delta_t, semantic_channels, timestep_shift_alpha))

class FluxGuidance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FluxGuidance",
            category="model/conditioning/flux",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Float.Input("guidance", default=3.5, min=0.0, max=100.0, step=0.1),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, conditioning, guidance) -> io.NodeOutput:
        c = node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})
        return io.NodeOutput(c)

    append = execute  # TODO: remove


class FluxDisableGuidance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FluxDisableGuidance",
            category="model/conditioning/flux",
            description="This node completely disables the guidance embed on Flux and Flux like models",
            inputs=[
                io.Conditioning.Input("conditioning"),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, conditioning) -> io.NodeOutput:
        c = node_helpers.conditioning_set_values(conditioning, {"guidance": None})
        return io.NodeOutput(c)

    append = execute  # TODO: remove


PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


class FluxKontextImageScale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FluxKontextImageScale",
            category="model/conditioning/flux",
            description="This node resizes the image to one that is more optimal for flux kontext.",
            inputs=[
                io.Image.Input("image"),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, image) -> io.NodeOutput:
        width = image.shape[2]
        height = image.shape[1]
        aspect_ratio = width / height
        _, width, height = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS)
        image = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)
        return io.NodeOutput(image)

    scale = execute  # TODO: remove


class FluxKontextMultiReferenceLatentMethod(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FluxKontextMultiReferenceLatentMethod",
            display_name="Edit Model Reference Method",
            category="model/conditioning/flux",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Combo.Input(
                    "reference_latents_method",
                    options=["offset", "index", "uxo/uno", "index_timestep_zero"],
                    advanced=True,
                ),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, conditioning, reference_latents_method) -> io.NodeOutput:
        if "uxo" in reference_latents_method or "uso" in reference_latents_method:
            reference_latents_method = "uxo"
        c = node_helpers.conditioning_set_values(conditioning, {"reference_latents_method": reference_latents_method})
        return io.NodeOutput(c)

    append = execute  # TODO: remove


def generalized_time_snr_shift(t, mu: float, sigma: float):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


def get_schedule(num_steps: int, image_seq_len: int) -> list[float]:
    mu = compute_empirical_mu(image_seq_len, num_steps)
    timesteps = torch.linspace(1, 0, num_steps + 1)
    timesteps = generalized_time_snr_shift(timesteps, mu, 1.0)
    return timesteps


class Flux2Scheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Flux2Scheduler",
            category="model/sampling/schedulers",
            inputs=[
                io.Int.Input("steps", default=20, min=1, max=4096),
                io.Int.Input("width", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=1),
                io.Int.Input("height", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=1),
            ],
            outputs=[
                io.Sigmas.Output(),
            ],
        )

    @classmethod
    def execute(cls, steps, width, height) -> io.NodeOutput:
        seq_len = (width * height / (16 * 16))
        sigmas = get_schedule(steps, round(seq_len))
        return io.NodeOutput(sigmas)

class KV_Attn_Input:
    def __init__(self):
        self.cache = {}

    def __call__(self, q, k, v, extra_options, **kwargs):
        reference_image_num_tokens = extra_options.get("reference_image_num_tokens", [])
        if len(reference_image_num_tokens) == 0:
            return {}

        ref_toks = sum(reference_image_num_tokens)
        cache_key = "{}_{}".format(extra_options["block_type"], extra_options["block_index"])
        if cache_key in self.cache:
            kk, vv = self.cache[cache_key]

            # Fix batch size changing.
            kk = comfy.utils.repeat_to_batch_size(kk, k.shape[0])
            vv = comfy.utils.repeat_to_batch_size(vv, v.shape[0])

            self.set_cache = False
            return {"q": q, "k": torch.cat((k, kk), dim=2), "v": torch.cat((v, vv), dim=2)}

        self.cache[cache_key] = (k[:, :, -ref_toks:].clone(), v[:, :, -ref_toks:].clone())
        self.set_cache = True
        return {"q": q, "k": k, "v": v}

    def cleanup(self):
        self.cache = {}


class FluxKVCache(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FluxKVCache",
            display_name="Flux KV Cache",
            description="Enables KV Cache optimization for reference images on Flux family models.",
            category="experimental",
            is_experimental=True,
            inputs=[
                io.Model.Input("model", tooltip="The model to use KV Cache on."),
            ],
            outputs=[
                io.Model.Output(tooltip="The patched model with KV Cache enabled."),
            ],
        )

    @classmethod
    def execute(cls, model: io.Model.Type) -> io.NodeOutput:
        m = model.clone()
        input_patch_obj = KV_Attn_Input()

        def model_input_patch(inputs):
            if len(input_patch_obj.cache) > 0:
                ref_image_tokens = sum(inputs["transformer_options"].get("reference_image_num_tokens", []))
                if ref_image_tokens > 0:
                    img = inputs["img"]
                    inputs["img"] = img[:, :-ref_image_tokens]
            return inputs

        m.set_model_attn1_patch(input_patch_obj)
        m.set_model_post_input_patch(model_input_patch)
        if hasattr(model.model.diffusion_model, "params"):
            m.add_object_patch("diffusion_model.params.default_ref_method", "index_timestep_zero")
        else:
            m.add_object_patch("diffusion_model.default_ref_method", "index_timestep_zero")

        return io.NodeOutput(m)

class FluxExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            CLIPTextEncodeFlux,
            FluxGuidance,
            FluxDisableGuidance,
            FluxKontextImageScale,
            FluxKontextMultiReferenceLatentMethod,
            EmptyFlux2LatentImage,
            EmptySefiLatent,
            SamplerSefiSemanticFirst,
            Flux2Scheduler,
            FluxKVCache,
        ]


async def comfy_entrypoint() -> FluxExtension:
    return FluxExtension()
