import logging
import math
from os import environ
from collections import OrderedDict

import gradio as gr
import modules.scripts as scripts
import torch

from modules import script_callbacks
from modules.script_callbacks import CFGDenoiserParams

try:
    from modules.rng import randn_like
except ImportError:  # Automatic1111 < 1.7 compatibility
    from torch import randn_like


logger = logging.getLogger(__name__)
logger.setLevel(environ.get("SD_WEBUI_LOG_LEVEL", logging.INFO))


def parse_bool(val, default=False):
    try:
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            val_lower = val.lower()
            if val_lower in ("true", "1", "yes", "on"):
                return True
            if val_lower in ("false", "0", "no", "off"):
                return False
        if isinstance(val, (int, float)):
            return bool(val)
    except Exception:
        pass
    return default


def parse_int(val, default=0):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def safe_randn_like(x, generator=None):
    """
    Safe randn_like wrapper that tolerates A1111/forge variants where modules.rng.randn_like
    may not support the generator kwarg.
    """
    try:
        if generator is not None:
            return randn_like(x, generator=generator)
        return randn_like(x)
    except TypeError:
        # Fallback to torch implementation if generator is passed but unsupported.
        if generator is not None:
            try:
                return torch.randn_like(x, generator=generator)
            except TypeError:
                return torch.randn_like(x)
        return torch.randn_like(x)

"""
An implementation of CADS: Unleashing the Diversity of Diffusion Models through
Condition-Annealed Sampling for Automatic1111 WebUI

@inproceedings{
    sadat2024cads,
    title={{CADS}: Unleashing the Diversity of Diffusion Models through Condition-Annealed Sampling},
    author={Seyedmorteza Sadat and Jakob Buhmann and Derek Bradley and Otmar Hilliges and Romann M. Weber},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=zMoNrajk2X}
}

Author: v0xie
GitHub URL: https://github.com/v0xie/sd-webui-cads
"""

STEP_RAMP_MODES = ("Hold after full", "Windowed 0→1", "Windowed 1→0")
DEFAULT_TAU1 = 0.6
DEFAULT_TAU2 = 0.9
DEFAULT_NOISE_SCALE = 0.25
DEFAULT_MIXING_FACTOR = 1.0
DEFAULT_SEED = -1
PRESETS = {
    "Subtle": {
        "tau1": 0.85,
        "tau2": 0.95,
        "noise_scale": 0.15,
        "mixing_factor": 1.0,
    },
    "Balanced": {
        "tau1": DEFAULT_TAU1,
        "tau2": DEFAULT_TAU2,
        "noise_scale": DEFAULT_NOISE_SCALE,
        "mixing_factor": DEFAULT_MIXING_FACTOR,
    },
    "Aggressive": {
        "tau1": 0.4,
        "tau2": 0.7,
        "noise_scale": 0.3,
        "mixing_factor": 0.8,
    },
}


class CADSExtensionScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.cads_generators = {}
        self.logged_unknown_conditioning = False

    # Extension title in menu UI
    def title(self):
        return "CADS"

    # Decide to show menu in txt2img or img2img
    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # Setup menu ui detail
    def ui(self, is_img2img):
        with gr.Accordion("CADS", open=False):
            active = gr.Checkbox(value=False, default=False, label="Active", elem_id="cads_active")
            rescale = gr.Checkbox(value=True, default=True, label="Rescale CFG", elem_id="cads_rescale")
            with gr.Row():
                use_step_mode = gr.Checkbox(
                    value=False,
                    default=False,
                    label="Use step sliders",
                    elem_id="cads_use_step_mode",
                    info="Convert Tau sliders to operate on absolute sampler steps instead of percentages.",
                )
                respect_strength = gr.Checkbox(
                    value=False,
                    default=False,
                    label="Scale steps by strength",
                    elem_id="cads_respect_strength",
                    info="When enabled, CADS step sliders account for denoising strength / t_enc for img2img and Hires. fix passes.",
                    interactive=False,
                )
                step_ramp_mode = gr.Dropdown(
                    value=STEP_RAMP_MODES[0],
                    choices=list(STEP_RAMP_MODES),
                    label="Step ramp mode",
                    elem_id="cads_step_ramp_mode",
                    info="How CADS strength behaves relative to the chosen step window.",
                    interactive=True,
                )
            with gr.Row():
                step_start = gr.Slider(
                    value=0,
                    minimum=0,
                    maximum=1000,
                    step=1,
                    label="Ramp start step (Tau 2)",
                    elem_id="cads_tau1_step",
                    info="Sampler step where CADS starts ramping in from 0 → 1 when using step sliders (corresponds to Tau 2 step).",
                    interactive=False,
                )
                step_stop = gr.Slider(
                    value=0,
                    minimum=0,
                    maximum=1000,
                    step=1,
                    label="Full strength step (Tau 1)",
                    elem_id="cads_tau2_step",
                    info="Sampler step where CADS reaches full strength before following the selected ramp mode (corresponds to Tau 1 step).",
                    interactive=False,
                )
            with gr.Row():
                t1 = gr.Slider(
                    value=DEFAULT_TAU1,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    label="Tau 1 (full strength point)",
                    elem_id="cads_tau1",
                    info="Normalized point where CADS reaches full strength; smaller values push CADS later in the schedule. Default 0.6",
                )
                t2 = gr.Slider(
                    value=DEFAULT_TAU2,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    label="Tau 2 (ramp start)",
                    elem_id="cads_tau2",
                    info="Normalized point where CADS starts ramping in from 0 → 1; higher values start earlier in the schedule. Default 0.9",
                )
            with gr.Row():
                noise_scale = gr.Slider(
                    value=DEFAULT_NOISE_SCALE,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    label="Noise Scale",
                    elem_id="cads_noise_scale",
                    info="Scale of noise injected at every time step, default 0.25, recommended <= 0.3",
                )
                mixing_factor = gr.Slider(
                    value=DEFAULT_MIXING_FACTOR,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    label="Mixing Factor",
                    elem_id="cads_mixing_factor",
                    info="Regularization factor, lowering this will increase the diversity of the images with more chance of divergence, default 1.0",
                )
            with gr.Accordion("Experimental", open=False):
                apply_to_hr_pass = gr.Checkbox(
                    value=False,
                    default=False,
                    label="Apply to Hires. Fix",
                    elem_id="cads_hr_fix_active",
                    info="Requires a very high denoising value to work. Default False",
                )
            with gr.Row():
                apply_to_positive = gr.Checkbox(
                    value=True,
                    default=True,
                    label="Apply to Positive",
                    elem_id="cads_apply_positive",
                )
                apply_to_negative = gr.Checkbox(
                    value=True,
                    default=True,
                    label="Apply to Negative",
                    elem_id="cads_apply_negative",
                )
            with gr.Row():
                presets = gr.Dropdown(
                    value="Balanced",
                    choices=list(PRESETS.keys()),
                    label="Presets",
                    elem_id="cads_preset",
                    info="Quickly load suggested Tau and noise settings.",
                )
                reset = gr.Button(value="Reset", elem_id="cads_reset")
            with gr.Row():
                cads_seed = gr.Number(
                    value=DEFAULT_SEED,
                    precision=0,
                    label="CADS Seed",
                    elem_id="cads_seed",
                    info="Seed used only for CADS noise (-1 follows main randomness).",
                )
                cads_seed_fixed = gr.Checkbox(
                    value=False,
                    default=False,
                    label="Fixed CADS Seed",
                    elem_id="cads_seed_fixed",
                    info="Use a dedicated RNG for CADS noise so it can be varied independently from the main seed.",
                )
            with gr.Row():
                same_noise_per_image = gr.Checkbox(
                    value=False,
                    default=False,
                    label="Same CADS noise across batch",
                    elem_id="cads_same_noise_per_image",
                    info="Requires a fixed CADS seed. When enabled, each image in a batch receives identical CADS noise per step (helpful for side-by-side comparisons).",
                    interactive=False,
                )
                share_noise_posneg = gr.Checkbox(
                    value=False,
                    default=False,
                    label="Share noise between pos/neg",
                    elem_id="cads_share_noise_posneg",
                    info="When enabled, positive and negative conditionings share the same CADS noise sample per step.",
                )

        def toggle_step_sliders(enabled):
            slider_enabled = gr.Slider.update(interactive=enabled)
            slider_disabled = gr.Slider.update(interactive=not enabled)
            checkbox_enabled = gr.Checkbox.update(interactive=enabled)
            dropdown_enabled = gr.Dropdown.update(interactive=True)
            return (
                slider_enabled,
                slider_enabled,
                slider_disabled,
                slider_disabled,
                checkbox_enabled,
                dropdown_enabled,
            )

        use_step_mode.change(
            fn=toggle_step_sliders,
            inputs=use_step_mode,
            outputs=[step_start, step_stop, t1, t2, respect_strength, step_ramp_mode],
        )

        def update_same_noise_enabled(seed_val, fixed):
            seed_int = parse_int(seed_val, DEFAULT_SEED)
            has_fixed_seed = seed_int >= 0 or parse_bool(fixed, False)
            info = "Requires a fixed CADS seed. When enabled, each image in a batch receives identical CADS noise per step (helpful for comparisons)."
            if has_fixed_seed:
                return gr.Checkbox.update(interactive=True, info=info)
            return gr.Checkbox.update(interactive=False, value=False, info=info)

        cads_seed.change(
            fn=update_same_noise_enabled,
            inputs=[cads_seed, cads_seed_fixed],
            outputs=same_noise_per_image,
        )
        cads_seed_fixed.change(
            fn=update_same_noise_enabled,
            inputs=[cads_seed, cads_seed_fixed],
            outputs=same_noise_per_image,
        )

        def apply_preset(preset_name):
            preset = PRESETS.get(preset_name, PRESETS["Balanced"])
            info = "Requires a fixed CADS seed. When enabled, each image in a batch receives identical CADS noise per step (helpful for comparisons)."
            return (
                gr.Slider.update(value=preset["tau2"]),
                gr.Slider.update(value=preset["tau1"]),
                gr.Slider.update(value=preset["noise_scale"]),
                gr.Slider.update(value=preset["mixing_factor"]),
                gr.Number.update(value=DEFAULT_SEED),
                gr.Checkbox.update(value=False),
                # Programmatic updates don't always trigger .change() handlers; keep UI consistent.
                gr.Checkbox.update(value=False, interactive=False, info=info),
            )

        presets.change(
            fn=apply_preset,
            inputs=presets,
            outputs=[t2, t1, noise_scale, mixing_factor, cads_seed, cads_seed_fixed, same_noise_per_image],
        )

        def reset_defaults():
            info = "Requires a fixed CADS seed. When enabled, each image in a batch receives identical CADS noise per step (helpful for comparisons)."
            return (
                gr.Slider.update(value=DEFAULT_TAU2),
                gr.Slider.update(value=DEFAULT_TAU1),
                gr.Slider.update(value=DEFAULT_NOISE_SCALE),
                gr.Slider.update(value=DEFAULT_MIXING_FACTOR),
                gr.Dropdown.update(value="Balanced"),
                gr.Number.update(value=DEFAULT_SEED),
                gr.Checkbox.update(value=False),
                # Programmatic updates don't always trigger .change() handlers; keep UI consistent.
                gr.Checkbox.update(value=False, interactive=False, info=info),
            )

        reset.click(
            fn=reset_defaults,
            inputs=None,
            outputs=[t2, t1, noise_scale, mixing_factor, presets, cads_seed, cads_seed_fixed, same_noise_per_image],
        )

        active.do_not_save_to_config = True
        rescale.do_not_save_to_config = True
        use_step_mode.do_not_save_to_config = True
        respect_strength.do_not_save_to_config = True
        step_ramp_mode.do_not_save_to_config = True
        step_start.do_not_save_to_config = True
        step_stop.do_not_save_to_config = True
        apply_to_hr_pass.do_not_save_to_config = True
        presets.do_not_save_to_config = True
        reset.do_not_save_to_config = True
        cads_seed_fixed.do_not_save_to_config = True
        same_noise_per_image.do_not_save_to_config = True
        share_noise_posneg.do_not_save_to_config = True

        # Keep Tau step infotext keys as legacy names for backward compatibility with existing images.
        self.infotext_fields = [
            (active, lambda d: gr.Checkbox.update(value=parse_bool(d.get("CADS Active", False), False))),
            (rescale, lambda d: gr.Checkbox.update(value=parse_bool(d.get("CADS Rescale", True), True))),
            (use_step_mode, lambda d: gr.Checkbox.update(value=parse_bool(d.get("CADS Use Step Mode", False), False))),
            (respect_strength, lambda d: gr.Checkbox.update(value=parse_bool(d.get("CADS Respect Strength", False), False))),
            (step_ramp_mode, lambda d: gr.Dropdown.update(value=d.get("CADS Step Ramp Mode", STEP_RAMP_MODES[0]) if d.get("CADS Step Ramp Mode", STEP_RAMP_MODES[0]) in STEP_RAMP_MODES else STEP_RAMP_MODES[0])),
            (step_start, "CADS Tau 1 Step"),
            (step_stop, "CADS Tau 2 Step"),
            (t1, "CADS Tau 1"),
            (t2, "CADS Tau 2"),
            (noise_scale, "CADS Noise Scale"),
            (mixing_factor, "CADS Mixing Factor"),
            (apply_to_hr_pass, "CADS Apply To Hires. Fix"),
            (apply_to_positive, lambda d: gr.Checkbox.update(value=parse_bool(d.get("CADS Apply To Positive", True), True))),
            (apply_to_negative, lambda d: gr.Checkbox.update(value=parse_bool(d.get("CADS Apply To Negative", True), True))),
            (presets, lambda d: gr.Dropdown.update(value=d.get("CADS Preset", "Balanced") if d.get("CADS Preset", "Balanced") in PRESETS else "Balanced")),
            (cads_seed, lambda d: gr.Number.update(value=parse_int(d.get("CADS Seed", DEFAULT_SEED), DEFAULT_SEED))),
            (cads_seed_fixed, lambda d: gr.Checkbox.update(value=parse_bool(d.get("CADS Fixed Seed", False), False))),
            (same_noise_per_image, lambda d: gr.Checkbox.update(value=parse_bool(d.get("CADS Same Noise Per Image", False), False))),
            (share_noise_posneg, lambda d: gr.Checkbox.update(value=parse_bool(d.get("CADS Share Noise Posneg", False), False))),
        ]
        self.paste_field_names = [
            "cads_active",
            "cads_rescale",
            "cads_use_step_mode",
            "cads_respect_strength",
            "cads_step_ramp_mode",
            "cads_tau1_step",
            "cads_tau2_step",
            "cads_tau1",
            "cads_tau2",
            "cads_noise_scale",
            "cads_mixing_factor",
            "cads_hr_fix_active",
            "cads_apply_positive",
            "cads_apply_negative",
            "cads_preset",
            "cads_seed",
            "cads_seed_fixed",
            "cads_same_noise_per_image",
            "cads_share_noise_posneg",
        ]
        return [
            active,
            use_step_mode,
            respect_strength,
            step_ramp_mode,
            step_start,
            step_stop,
            t1,
            t2,
            noise_scale,
            mixing_factor,
            rescale,
            apply_to_hr_pass,
            apply_to_positive,
            apply_to_negative,
            presets,
            cads_seed,
            cads_seed_fixed,
            same_noise_per_image,
            share_noise_posneg,
        ]

    def before_process_batch(
        self,
        p,
        active,
        use_step_mode,
        respect_strength,
        step_ramp_mode,
        step_start,
        step_stop,
        t1,
        t2,
        noise_scale,
        mixing_factor,
        rescale,
        apply_to_hr_pass,
        apply_to_positive,
        apply_to_negative,
        presets,
        cads_seed,
        cads_seed_fixed,
        same_noise_per_image,
        share_noise_posneg,
        *args,
        **kwargs,
    ):
        self.unhook_callbacks()
        self.cads_generators = {}
        self.logged_unknown_conditioning = False
        self.logged_same_noise_warning = False
        active = parse_bool(getattr(p, "cads_active", active), False)
        if not active:
            return

        use_step_mode = parse_bool(getattr(p, "cads_use_step_mode", use_step_mode), use_step_mode)
        respect_strength = parse_bool(getattr(p, "cads_respect_strength", respect_strength), respect_strength)
        ramp_mode_raw = getattr(p, "cads_step_ramp_mode", step_ramp_mode) or STEP_RAMP_MODES[0]
        ramp_mode = ramp_mode_raw if ramp_mode_raw in STEP_RAMP_MODES else STEP_RAMP_MODES[0]
        step_start = parse_int(getattr(p, "cads_tau1_step", step_start), step_start)
        step_stop  = parse_int(getattr(p, "cads_tau2_step", step_stop),  step_stop)
        steps = getattr(p, "steps", -1)
        if use_step_mode and steps <= 0:
            logger.error("Steps not set, disabling CADS step sliders")
            use_step_mode = False

        strength_scale = 1.0
        effective_steps_float = float(max(steps, 1))
        effective_step_count = max(int(round(effective_steps_float)), 1)
        if use_step_mode:
            step_start = int(max(min(step_start, steps), 0))
            step_stop = int(max(min(step_stop, steps), 0))
            if step_stop < step_start:
                step_stop = step_start
            steps_float = float(max(steps, 1))
            if respect_strength:
                strength_raw = getattr(p, "denoising_strength", None)
                try:
                    strength_scale = float(strength_raw)
                except (TypeError, ValueError):
                    strength_scale = 1.0
                if not math.isfinite(strength_scale):
                    strength_scale = 1.0
                strength_scale = max(min(strength_scale, 1.0), 0.0)
                if strength_scale == 0.0:
                    logger.warning("CADS: Denoising strength is 0.0, falling back to 1.0 for step conversion")
                    strength_scale = 1.0
                effective_steps_float = max(steps_float * strength_scale, 1.0)
                effective_step_count = max(int(round(effective_steps_float)), 1)
                step_start = int(max(min(step_start, effective_step_count), 0))
                step_stop = int(max(min(step_stop, effective_step_count), 0))
                if step_stop < step_start:
                    step_stop = step_start
                steps_float = effective_steps_float
            t1 = max(min(1.0 - (step_stop / steps_float), 1.0), 0.0)
            t2 = max(min(1.0 - (step_start / steps_float), 1.0), 0.0)
        else:
            step_start = 0
            step_stop = 0
            t1 = getattr(p, "cads_tau1", t1)
            t2 = getattr(p, "cads_tau2", t2)
            try:
                t1 = float(t1)
            except (TypeError, ValueError):
                t1 = DEFAULT_TAU1
            try:
                t2 = float(t2)
            except (TypeError, ValueError):
                t2 = DEFAULT_TAU2
            strength_scale = 1.0
            effective_step_count = max(int(round(float(max(steps, 1)))), 1)

        noise_scale = getattr(p, "cads_noise_scale", noise_scale)
        mixing_factor = getattr(p, "cads_mixing_factor", mixing_factor)
        rescale = parse_bool(getattr(p, "cads_rescale", rescale), rescale)
        try:
            noise_scale = max(min(float(noise_scale), 1.0), 0.0)
        except (TypeError, ValueError):
            noise_scale = DEFAULT_NOISE_SCALE
        try:
            mixing_factor = max(min(float(mixing_factor), 1.0), 0.0)
        except (TypeError, ValueError):
            mixing_factor = DEFAULT_MIXING_FACTOR
        apply_to_hr_pass = parse_bool(getattr(p, "cads_hr_fix_active", apply_to_hr_pass), apply_to_hr_pass)
        apply_to_positive = parse_bool(getattr(p, "cads_apply_positive", apply_to_positive), apply_to_positive)
        apply_to_negative = parse_bool(getattr(p, "cads_apply_negative", apply_to_negative), apply_to_negative)
        preset_name_raw = getattr(p, "cads_preset", presets)
        preset_name = preset_name_raw if preset_name_raw in PRESETS else "Balanced"
        cads_seed_value = parse_int(getattr(p, "cads_seed", cads_seed), DEFAULT_SEED)
        cads_seed_fixed = parse_bool(getattr(p, "cads_seed_fixed", cads_seed_fixed), False)
        same_noise_per_image = parse_bool(getattr(p, "cads_same_noise_per_image", same_noise_per_image), False)
        share_noise_posneg = parse_bool(getattr(p, "cads_share_noise_posneg", share_noise_posneg), False)
        generator_seed = None
        if cads_seed_value >= 0:
            generator_seed = cads_seed_value
        elif cads_seed_fixed:
            base_seed = getattr(p, "seed", None)
            if base_seed is None:
                all_seeds = getattr(p, "all_seeds", None)
                if isinstance(all_seeds, (list, tuple)) and all_seeds:
                    base_seed = all_seeds[0]
            if base_seed == -1:
                # Treat a random main seed (-1) as “no fixed CADS seed”; fall back to nondeterministic CADS noise.
                generator_seed = None
            else:
                if base_seed is None:
                    base_seed = torch.seed()
                try:
                    generator_seed = int(base_seed)
                except (TypeError, ValueError):
                    generator_seed = DEFAULT_SEED
        if generator_seed is not None and generator_seed < 0:
            generator_seed = None
        self.cads_generator_seed = generator_seed if generator_seed is not None and generator_seed >= 0 else None

        first_pass_steps = getattr(p, "steps", -1)
        if first_pass_steps <= 0:
            logger.error("Steps not set, disabling CADS")
            return

        setattr(p, "cads_use_step_mode", use_step_mode)
        setattr(p, "cads_respect_strength", respect_strength)
        setattr(p, "cads_step_ramp_mode", ramp_mode)
        setattr(p, "cads_tau1_step", step_start)
        setattr(p, "cads_tau2_step", step_stop)
        setattr(p, "cads_tau1", t1)
        setattr(p, "cads_tau2", t2)
        setattr(p, "cads_strength_scale", strength_scale)
        setattr(p, "cads_effective_steps", effective_step_count)
        setattr(p, "cads_apply_positive", apply_to_positive)
        setattr(p, "cads_apply_negative", apply_to_negative)
        setattr(p, "cads_preset", preset_name)
        setattr(p, "cads_seed", cads_seed_value)
        setattr(p, "cads_seed_fixed", cads_seed_fixed)
        setattr(p, "cads_same_noise_per_image", same_noise_per_image)
        setattr(p, "cads_share_noise_posneg", share_noise_posneg)

        if not hasattr(p, "extra_generation_params") or not isinstance(p.extra_generation_params, dict):
            p.extra_generation_params = {}

        gen_params = {
            "CADS Active": active,
            "CADS Use Step Mode": use_step_mode,
            "CADS Respect Strength": respect_strength,
            "CADS Step Ramp Mode": ramp_mode if ramp_mode in STEP_RAMP_MODES else STEP_RAMP_MODES[0],
            "CADS Tau 1 Step": step_start,
            "CADS Tau 2 Step": step_stop,
            "CADS Tau 1": t1,
            "CADS Tau 2": t2,
            "CADS Noise Scale": noise_scale,
            "CADS Mixing Factor": mixing_factor,
            "CADS Rescale": rescale,
            "CADS Apply To Hires. Fix": apply_to_hr_pass,
            "CADS Strength Scale": strength_scale,
            "CADS Effective Steps": effective_step_count,
            "CADS Apply To Positive": apply_to_positive,
            "CADS Apply To Negative": apply_to_negative,
            "CADS Preset": preset_name if preset_name in PRESETS else "Balanced",
            "CADS Seed": cads_seed_value,
            "CADS Fixed Seed": cads_seed_fixed,
            "CADS Same Noise Per Image": same_noise_per_image,
            "CADS Share Noise Posneg": share_noise_posneg,
        }
        if self.cads_generator_seed is not None:
            gen_params["CADS Effective Seed"] = self.cads_generator_seed

        p.extra_generation_params.update(gen_params)

        self.create_hook(
            p,
            active,
            t1,
            t2,
            noise_scale,
            mixing_factor,
            rescale,
            effective_step_count,
            ramp_mode,
            apply_to_positive,
            apply_to_negative,
            preset_name,
            self.cads_generator_seed,
            same_noise_per_image,
            share_noise_posneg,
        )

    def create_hook(
        self,
        p,
        active,
        t1,
        t2,
        noise_scale,
        mixing_factor,
        rescale,
        total_sampling_steps,
        ramp_mode,
        apply_to_positive,
        apply_to_negative,
        preset_name,
        generator_seed,
        same_noise_per_image,
        share_noise_posneg,
    ):
        callback = lambda params: self.on_cfg_denoiser_callback(
            params,
            t1=t1,
            t2=t2,
            noise_scale=noise_scale,
            mixing_factor=mixing_factor,
            rescale=rescale,
            total_sampling_steps=total_sampling_steps,
            ramp_mode=ramp_mode,
            apply_to_positive=apply_to_positive,
            apply_to_negative=apply_to_negative,
            generator_seed=generator_seed,
            same_noise_per_image=same_noise_per_image,
            share_noise_posneg=share_noise_posneg,
        )

        logger.debug("Hooked callbacks")
        script_callbacks.on_cfg_denoiser(callback)
        script_callbacks.on_script_unloaded(self.unhook_callbacks)

    def postprocess_batch(self, p, *args, **kwargs):
        self.unhook_callbacks()

    def unhook_callbacks(self):
        logger.debug("Unhooked callbacks")
        script_callbacks.remove_current_script_callbacks()

    def cads_linear_schedule(self, t, tau1, tau2, mode):
        """CADS annealing schedule function."""
        tau1 = float(max(min(tau1, 1.0), 0.0))
        tau2 = float(max(min(tau2, 1.0), 0.0))

        if tau1 >= tau2:
            if mode == "Hold after full":
                return 1.0 if t <= tau1 else 0.0
            return 0.0

        if t >= tau2:
            return 0.0
        if t <= tau1:
            return 1.0 if mode == "Hold after full" else 0.0

        denom = max(tau2 - tau1, 1e-8)
        gamma_up = (tau2 - t) / denom
        gamma_up = 0.0 if gamma_up < 0.0 else (1.0 if gamma_up > 1.0 else gamma_up)

        if mode == "Windowed 1→0":
            return 1.0 - gamma_up
        # Default to Hold-after / Windowed 0→1 behaviour
        return gamma_up

    def add_noise(self, y, gamma, noise_scale, psi, rescale=False, generator=None, noise_override=None):
        """CADS adding noise to the condition."""
        gamma = max(min(float(gamma), 1.0), 0.0)
        base = math.sqrt(gamma)
        residual = math.sqrt(max(1.0 - gamma, 0.0))
        psi = max(min(float(psi), 1.0), 0.0)
        noise_scale = max(min(float(noise_scale), 1.0), 0.0)
        reduce_dims = tuple(range(1, y.dim())) if y.dim() > 1 else None
        y_mean = torch.mean(y, dim=reduce_dims, keepdim=reduce_dims is not None)
        y_std = torch.std(y, dim=reduce_dims, keepdim=reduce_dims is not None, unbiased=False)
        if noise_override is not None:
            noise = noise_override
        else:
            noise = safe_randn_like(y, generator=generator)
        y = base * y + noise_scale * residual * noise
        if rescale:
            denom = torch.std(y, dim=reduce_dims, keepdim=reduce_dims is not None, unbiased=False)
            if torch.any(denom == 0):
                logger.debug("Warning: zero standard deviation encountered during rescaling")
            denom = denom.clamp_min(1e-8)
            y_scaled = (y - torch.mean(y, dim=reduce_dims, keepdim=reduce_dims is not None)) / denom * y_std + y_mean
            if torch.isfinite(y_scaled).all():
                y = psi * y_scaled + (1 - psi) * y
            else:
                logger.debug("Warning: non-finite encountered in rescaling")
        return y

    def on_cfg_denoiser_callback(
        self,
        params: CFGDenoiserParams,
        t1,
        t2,
        noise_scale,
        mixing_factor,
        rescale,
        total_sampling_steps,
        ramp_mode,
        apply_to_positive,
        apply_to_negative,
        generator_seed,
        same_noise_per_image,
        share_noise_posneg,
    ):
        sampling_step = params.sampling_step
        total_sampling_step = max(int(total_sampling_steps), 1)
        text_cond = params.text_cond
        text_uncond = params.text_uncond

        t = 1.0 - max(min((sampling_step + 1) / total_sampling_step, 1.0), 0.0)
        gamma = self.cads_linear_schedule(t, t1, t2, mode=ramp_mode)

        def get_generator(device):
            if generator_seed is None:
                return None
            key = str(device)
            existing = self.cads_generators.get(key)
            if existing and existing["seed"] == generator_seed and existing["device"] == device:
                return existing["generator"]
            gen = torch.Generator(device=device)
            gen.manual_seed(generator_seed)
            self.cads_generators[key] = {"generator": gen, "seed": generator_seed, "device": device}
            return gen

        if same_noise_per_image and generator_seed is None and not self.logged_same_noise_warning:
            logger.warning("CADS: Same noise across batch requested but no fixed CADS seed available; disabling shared noise for this run")
            self.logged_same_noise_warning = True
            same_noise_per_image = False

        def shared_noise_for_tensor(tensor, gen):
            """Returns noise matching tensor shape for batch images using a shared generator state."""
            if not same_noise_per_image or generator_seed is None:
                return None
            if tensor.dim() < 1 or tensor.shape[0] < 1 or gen is None:
                return None
            base_noise = safe_randn_like(tensor[0], generator=gen)
            # Broadcast base noise across batch while preserving non-batch dimensions.
            return base_noise.unsqueeze(0).expand((tensor.shape[0],) + base_noise.shape)

        if isinstance(text_cond, torch.Tensor) and isinstance(text_uncond, torch.Tensor):
            gen = get_generator(text_cond.device)
            shared_noise_pos = shared_noise_for_tensor(text_cond, gen) if apply_to_positive else None
            if apply_to_positive:
                params.text_cond = self.add_noise(
                    text_cond, gamma, noise_scale, mixing_factor, rescale, gen, shared_noise_pos
                )
            if apply_to_negative:
                shared_noise_neg = shared_noise_pos if share_noise_posneg else shared_noise_for_tensor(text_uncond, gen)
                params.text_uncond = self.add_noise(
                    text_uncond, gamma, noise_scale, mixing_factor, rescale, gen, shared_noise_neg
                )
        elif isinstance(text_cond, (dict, OrderedDict)) and isinstance(text_uncond, (dict, OrderedDict)):
            for key in ("crossattn", "vector"):
                if key in text_cond and key in text_uncond:
                    v = text_cond[key]
                    u = text_uncond[key]
                    if isinstance(v, torch.Tensor) and isinstance(u, torch.Tensor):
                        gen = get_generator(v.device)
                        shared_noise_pos = shared_noise_for_tensor(v, gen) if apply_to_positive else None
                        if apply_to_positive:
                            params.text_cond[key] = self.add_noise(
                                v, gamma, noise_scale, mixing_factor, rescale, gen, shared_noise_pos
                            )
                        if apply_to_negative:
                            shared_noise_neg = shared_noise_pos if share_noise_posneg else shared_noise_for_tensor(u, gen)
                            params.text_uncond[key] = self.add_noise(
                                u, gamma, noise_scale, mixing_factor, rescale, gen, shared_noise_neg
                            )
        else:
            if not self.logged_unknown_conditioning:
                typename = type(text_cond).__name__
                keys = list(text_cond.keys()) if isinstance(text_cond, dict) else None
                logger.warning("CADS: Unknown text_cond type (%s) keys=%s; skipping CADS noise", typename, keys)
                self.logged_unknown_conditioning = True

    def before_hr(self, p, *args):
        self.unhook_callbacks()
        self.cads_generators = {}
        self.logged_unknown_conditioning = False
        self.logged_same_noise_warning = False

        params = getattr(p, "extra_generation_params", None)
        if not params:
            logger.error("Missing attribute extra_generation_params")
            return

        active = parse_bool(params.get("CADS Active", False), False)
        if not active:
            return

        apply_to_hr_pass = parse_bool(params.get("CADS Apply To Hires. Fix", False), False)
        if not apply_to_hr_pass:
            logger.debug("Disabled for hires. fix")
            return

        # Infotext params may arrive as strings; normalize types.
        t1 = params.get("CADS Tau 1", None)
        t2 = params.get("CADS Tau 2", None)
        noise_scale = params.get("CADS Noise Scale", None)
        mixing_factor = params.get("CADS Mixing Factor", None)
        rescale = parse_bool(params.get("CADS Rescale", True), True)
        use_step_mode = parse_bool(params.get("CADS Use Step Mode", False), False)
        respect_strength = parse_bool(params.get("CADS Respect Strength", False), False)

        try:
            t1 = float(t1)
        except (TypeError, ValueError):
            t1 = None
        try:
            t2 = float(t2)
        except (TypeError, ValueError):
            t2 = None
        try:
            noise_scale = float(noise_scale)
        except (TypeError, ValueError):
            noise_scale = None
        try:
            mixing_factor = float(mixing_factor)
        except (TypeError, ValueError):
            mixing_factor = None

        if noise_scale is not None:
            noise_scale = max(min(noise_scale, 1.0), 0.0)
        if mixing_factor is not None:
            mixing_factor = max(min(mixing_factor, 1.0), 0.0)


        ramp_mode = params.get("CADS Step Ramp Mode", STEP_RAMP_MODES[0])
        if ramp_mode not in STEP_RAMP_MODES:
            ramp_mode = STEP_RAMP_MODES[0]
        step_start = parse_int(params.get("CADS Tau 1 Step", 0), 0)
        step_stop  = parse_int(params.get("CADS Tau 2 Step", 0), 0)
        apply_to_positive = parse_bool(params.get("CADS Apply To Positive", True), True)
        apply_to_negative = parse_bool(params.get("CADS Apply To Negative", True), True)
        preset_name_raw = params.get("CADS Preset", "Balanced")
        preset_name = preset_name_raw if preset_name_raw in PRESETS else "Balanced"
        same_noise_per_image = parse_bool(params.get("CADS Same Noise Per Image", False), False)
        share_noise_posneg = parse_bool(params.get("CADS Share Noise Posneg", False), False)
        generator_seed = parse_int(params.get("CADS Seed", None), None)
        generator_seed = generator_seed if generator_seed is None or generator_seed >= 0 else None
        effective_seed = params.get("CADS Effective Seed", None)
        effective_seed = parse_int(effective_seed, None)
        if effective_seed is not None and effective_seed < 0:
            effective_seed = None
        if effective_seed is None:
            if generator_seed is None and parse_bool(params.get("CADS Fixed Seed", False), False) and parse_int(params.get("CADS Seed", DEFAULT_SEED), DEFAULT_SEED) < 0:
                base_seed = getattr(p, "seed", None)
                if base_seed is None:
                    all_seeds = getattr(p, "all_seeds", None)
                    if isinstance(all_seeds, (list, tuple)) and all_seeds:
                        base_seed = all_seeds[0]
                if base_seed == -1:
                    effective_seed = None
                else:
                    try:
                        effective_seed = int(base_seed)
                    except (TypeError, ValueError):
                        effective_seed = None
            else:
                effective_seed = generator_seed
        generator_seed = effective_seed

        if None in (t1, t2, noise_scale, mixing_factor, rescale):
            logger.error("Missing needed parameters for Hires. fix")
            return

        hr_pass_steps = getattr(p, "hr_second_pass_steps", -1)
        if hr_pass_steps < 0:
            logger.error("Attribute hr_second_pass_steps not found")
            return
        if hr_pass_steps == 0:
            logger.debug("Using first pass step count for hires. fix")
            hr_pass_steps = getattr(p, "steps", -1)

        effective_step_count = params.get(
            "CADS Effective Steps", max(int(round(float(max(hr_pass_steps, 1)))), 1)
        )
        if use_step_mode and hr_pass_steps > 0:
            steps_float = float(max(hr_pass_steps, 1))
            if respect_strength:
                strength_raw = getattr(p, "hr_denoising_strength", getattr(p, "denoising_strength", None))
                try:
                    strength_scale = float(strength_raw)
                except (TypeError, ValueError):
                    strength_scale = 1.0
                if not math.isfinite(strength_scale):
                    strength_scale = 1.0
                strength_scale = max(min(strength_scale, 1.0), 0.0)
                if strength_scale == 0.0:
                    logger.warning(
                        "CADS: Hires. fix denoising strength is 0.0, falling back to 1.0 for step conversion"
                    )
                    strength_scale = 1.0
                steps_float = max(steps_float * strength_scale, 1.0)
                params["CADS Strength Scale"] = strength_scale
            effective_step_count = max(int(round(steps_float)), 1)
            step_start = int(max(min(step_start, effective_step_count), 0))
            step_stop = int(max(min(step_stop, effective_step_count), 0))
            if step_stop < step_start:
                step_stop = step_start
            t1 = max(min(1.0 - (step_stop / steps_float), 1.0), 0.0)
            t2 = max(min(1.0 - (step_start / steps_float), 1.0), 0.0)
            params["CADS Tau 1 Step"] = step_start
            params["CADS Tau 2 Step"] = step_stop
            params["CADS Tau 1"] = t1
            params["CADS Tau 2"] = t2
            params["CADS Effective Steps"] = effective_step_count

        logger.debug("Enabled for hi-res fix with %i steps, re-hooking CADS", hr_pass_steps)
        total_sampling_steps = (
            params.get("CADS Effective Steps", effective_step_count)
            if use_step_mode and respect_strength
            else hr_pass_steps
        )
        self.create_hook(
            p,
            active,
            t1,
            t2,
            noise_scale,
            mixing_factor,
            rescale,
            total_sampling_steps,
            ramp_mode,
            apply_to_positive,
            apply_to_negative,
            preset_name,
            generator_seed,
            same_noise_per_image,
            share_noise_posneg,
        )


def cads_apply_override(field, boolean: bool = False):
    def fun(p, x, xs):
        if boolean:
            x = parse_bool(x, False)
        setattr(p, field, x)

    return fun


def cads_apply_field(field):
    def fun(p, x, xs):
        if not hasattr(p, "cads_active"):
            setattr(p, "cads_active", True)
        setattr(p, field, x)

    return fun


def cads_apply_step_field(field):
    def fun(p, x, xs):
        if not hasattr(p, "cads_active"):
            setattr(p, "cads_active", True)
        setattr(p, "cads_use_step_mode", True)
        setattr(p, field, x)

    return fun


def make_axis_options():
    xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ in ("xyz_grid.py", "scripts.xyz_grid")][0].module
    if not hasattr(xyz_grid, "boolean_choice"):
        xyz_grid.boolean_choice = lambda reverse=False: ["True", "False"] if not reverse else ["False", "True"]
    if not hasattr(xyz_grid, "choice_step_ramp_mode"):
        xyz_grid.choice_step_ramp_mode = lambda: list(STEP_RAMP_MODES)

    extra_axis_options = [
        xyz_grid.AxisOption("[CADS] Active", str, cads_apply_override("cads_active", boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
        xyz_grid.AxisOption("[CADS] Rescale CFG", str, cads_apply_override("cads_rescale", boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
        xyz_grid.AxisOption("[CADS] Use Step Mode", str, cads_apply_override("cads_use_step_mode", boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
        xyz_grid.AxisOption("[CADS] Respect Strength", str, cads_apply_override("cads_respect_strength", boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
        xyz_grid.AxisOption("[CADS] Step Ramp Mode", str, cads_apply_field("cads_step_ramp_mode"), choices=xyz_grid.choice_step_ramp_mode()),
        xyz_grid.AxisOption("[CADS] Tau 1 Step", int, cads_apply_step_field("cads_tau1_step")),
        xyz_grid.AxisOption("[CADS] Tau 2 Step", int, cads_apply_step_field("cads_tau2_step")),
        xyz_grid.AxisOption("[CADS] Tau 1", float, cads_apply_field("cads_tau1")),
        xyz_grid.AxisOption("[CADS] Tau 2", float, cads_apply_field("cads_tau2")),
        xyz_grid.AxisOption("[CADS] Noise Scale", float, cads_apply_field("cads_noise_scale")),
        xyz_grid.AxisOption("[CADS] Mixing Factor", float, cads_apply_field("cads_mixing_factor")),
        xyz_grid.AxisOption("[CADS] Apply to Hires. Fix", str, cads_apply_override("cads_hr_fix_active", boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
        xyz_grid.AxisOption("[CADS] Apply to Positive", str, cads_apply_override("cads_apply_positive", boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
        xyz_grid.AxisOption("[CADS] Apply to Negative", str, cads_apply_override("cads_apply_negative", boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
        xyz_grid.AxisOption("[CADS] CADS Seed", int, cads_apply_field("cads_seed")),
        xyz_grid.AxisOption("[CADS] Fixed CADS Seed", str, cads_apply_override("cads_seed_fixed", boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
        xyz_grid.AxisOption("[CADS] Same CADS noise across batch", str, cads_apply_override("cads_same_noise_per_image", boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
        xyz_grid.AxisOption("[CADS] Share noise between pos/neg", str, cads_apply_override("cads_share_noise_posneg", boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
    ]
    if not any("[CADS]" in x.label for x in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(extra_axis_options)


def callback_before_ui():
    try:
        make_axis_options()
    except Exception:
        logger.exception("CADS: Error while making axis options")


script_callbacks.on_before_ui(callback_before_ui)
