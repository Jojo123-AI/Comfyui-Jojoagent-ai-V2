# Comfyui-Jojoagent-ai-V2 v2026.05.27

## Fixes
- Fixed RH image generation when image ratio is set to 智能比例 by converting it to a concrete supported ratio before request submission.
- Fixed Jojo V2 提示词优化器 prompts_list output so batch image generation can split multi-screen prompts reliably.
- Standardized prompt list output as quoted blocks beginning with 第N屏提示词： and separated by blank lines.
- Kept Jojo V2 电商详情页提示词 prompt list output consistent with the optimizer node.

## Install
Copy the Comfyui-Jojoagent-ai-V2 folder into ComfyUI/custom_nodes/ and restart ComfyUI.
