import numpy as np
from moviepy import Clip, ColorClip, CompositeVideoClip, vfx
from PIL import Image, ImageEnhance


# FadeIn
def fadein_transition(clip: Clip, t: float) -> Clip:
    return clip.with_effects([vfx.FadeIn(t)])


# FadeOut
def fadeout_transition(clip: Clip, t: float) -> Clip:
    return clip.with_effects([vfx.FadeOut(t)])


# SlideIn
def slidein_transition(clip: Clip, t: float, side: str) -> Clip:
    width, height = clip.size

    # MoviePy 内置 SlideIn 在当前这条处理链里对全屏素材不稳定，
    # 会出现“逻辑上应用了转场，但画面几乎看不出变化”的情况。
    # 这里改成显式黑底 + 位移动画，保证转场效果可见且行为可控。
    def position(current_time: float):
        progress = min(max(current_time / max(t, 0.001), 0), 1)

        if side == "left":
            return (-width + width * progress, 0)
        if side == "right":
            return (width - width * progress, 0)
        if side == "top":
            return (0, -height + height * progress)
        if side == "bottom":
            return (0, height - height * progress)
        return (0, 0)

    background = ColorClip(size=(width, height), color=(0, 0, 0)).with_duration(
        clip.duration
    )
    moving_clip = clip.with_position(position)
    return CompositeVideoClip([background, moving_clip], size=(width, height)).with_duration(
        clip.duration
    )


# SlideOut
def slideout_transition(clip: Clip, t: float, side: str) -> Clip:
    width, height = clip.size
    transition_start = max(clip.duration - t, 0)

    # SlideOut 同样改成显式位移，保证片段末尾能稳定滑出画面。
    def position(current_time: float):
        if current_time <= transition_start:
            return (0, 0)

        progress = min(
            max((current_time - transition_start) / max(t, 0.001), 0), 1
        )

        if side == "left":
            return (-width * progress, 0)
        if side == "right":
            return (width * progress, 0)
        if side == "top":
            return (0, -height * progress)
        if side == "bottom":
            return (0, height * progress)
        return (0, 0)

    background = ColorClip(size=(width, height), color=(0, 0, 0)).with_duration(
        clip.duration
    )
    moving_clip = clip.with_position(position)
    return CompositeVideoClip([background, moving_clip], size=(width, height)).with_duration(
        clip.duration
    )


# 保留原始设计的 20% 缩放幅度，让三秒左右的短片也有清晰可见的 Ken Burns 运动感。
# 缩放稳定性由下方的亚像素中心采样保证，不通过削弱效果幅度来掩盖源视频编码闪烁。
_ZOOM_MAX_SCALE = 1.2


def _zoom_frame(frame: np.ndarray, scale_factor: float) -> np.ndarray:
    """使用亚像素中心裁剪实现无黑边且稳定的缩放效果。

    不能先把裁剪宽高转换为整数：缩放比例连续变化时，整数边界会按不同步长跳动，
    并在奇偶尺寸切换时改变半像素采样相位，最终表现为画面抖动。Pillow 的 EXTENT
    变换可以直接接收浮点边界，在固定输出画布上完成亚像素采样；左右、上下边界
    始终围绕同一个浮点中心对称，因此适用于整段视频持续缓慢缩放的场景。
    """
    if scale_factor <= 0:
        raise ValueError("scale_factor must be greater than zero")

    # 1 倍缩放直接返回原帧，避免无意义的重采样造成首帧轻微模糊。
    if abs(scale_factor - 1.0) < 1e-9:
        return frame

    height, width = frame.shape[:2]
    crop_width = width / scale_factor
    crop_height = height / scale_factor
    left = (width - crop_width) / 2
    top = (height - crop_height) / 2
    right = left + crop_width
    bottom = top + crop_height

    image = Image.fromarray(frame)
    transformed = image.transform(
        (width, height),
        Image.Transform.EXTENT,
        (left, top, right, bottom),
        # 视频连续缩放更关注相邻帧的一致性。BICUBIC/LANCZOS 虽然单帧更锐利，
        # 但高频纹理跨越采样网格时容易出现振铃和亮度闪烁；BILINEAR 更柔和，
        # 能以少量锐度损失换取更稳定的动态观感。
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(transformed)


def zoomin_transition(clip: Clip, t: float) -> Clip:
    """在整个片段内从原始画面平滑放大到 1.2 倍。"""
    # t 暂时保留，用于与其它转场函数保持统一调用签名；缩放需要覆盖完整片段，
    # 否则短暂缩放结束后画面会突然静止，不适合静态或低运动量素材。
    _ = t
    duration = max(clip.duration, 0.001)

    def scale_effect(get_frame, current_time: float):
        progress = min(max(current_time / duration, 0), 1)
        scale_factor = 1 + (_ZOOM_MAX_SCALE - 1) * progress
        return _zoom_frame(get_frame(current_time), scale_factor)

    return clip.transform(scale_effect)


def zoomout_transition(clip: Clip, t: float) -> Clip:
    """在整个片段内从 1.2 倍平滑缩小到原始画面。"""
    # 与 zoomin_transition 一致，t 仅用于兼容统一的转场调用接口。
    _ = t
    duration = max(clip.duration, 0.001)

    def scale_effect(get_frame, current_time: float):
        progress = min(max(current_time / duration, 0), 1)
        scale_factor = _ZOOM_MAX_SCALE - (_ZOOM_MAX_SCALE - 1) * progress
        return _zoom_frame(get_frame(current_time), scale_factor)

    return clip.transform(scale_effect)


# Cinematic color grade: mild contrast/saturation boost plus a soft vignette.
# Vignette masks are cached per-resolution since stock clips share a handful
# of output sizes and recomputing the radial gradient every frame is wasted
# work at 30-60fps.
_VIGNETTE_CACHE: dict = {}


def _vignette_mask(width: int, height: int) -> np.ndarray:
    key = (width, height)
    cached = _VIGNETTE_CACHE.get(key)
    if cached is not None:
        return cached

    y, x = np.ogrid[:height, :width]
    cx, cy = width / 2, height / 2
    max_dist = (cx**2 + cy**2) ** 0.5
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_dist
    # Keep the center untouched; only darken the outer ~45% toward the edges.
    mask = 1 - np.clip((dist - 0.55) / 0.45, 0, 1) * 0.35
    mask = mask.astype(np.float32)
    _VIGNETTE_CACHE[key] = mask
    return mask


def _apply_cinematic_grade(frame: np.ndarray) -> np.ndarray:
    # Degenerate frames (near-zero size, float dtype from an upstream
    # transform) can't go through PIL's Image.fromarray. Rather than abort
    # the whole clip over a handful of bad frames, pass those through
    # unmodified — a few ungraded frames are invisible; a dropped clip isn't.
    if frame.dtype != np.uint8 or frame.ndim != 3 or min(frame.shape[:2]) < 2:
        return frame

    try:
        image = Image.fromarray(frame)
        image = ImageEnhance.Contrast(image).enhance(1.12)
        image = ImageEnhance.Color(image).enhance(1.15)
        image = ImageEnhance.Brightness(image).enhance(0.97)
        graded = np.asarray(image).astype(np.float32)

        height, width = graded.shape[:2]
        mask = _vignette_mask(width, height)
        graded *= mask[..., None]
        return np.clip(graded, 0, 255).astype(np.uint8)
    except Exception:
        return frame


def cinematic_grade(clip: Clip) -> Clip:
    """Apply a subtle cinematic color grade (contrast/saturation/vignette)."""

    def grade_effect(get_frame, current_time: float):
        return _apply_cinematic_grade(get_frame(current_time))

    return clip.transform(grade_effect)
