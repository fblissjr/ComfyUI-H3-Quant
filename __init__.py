"""ComfyUI custom node: MiniMax H3 channel balance for INT8 attention. Off by default."""

from .channel_balance import MiniMaxH3ChannelBalance

NODE_CLASS_MAPPINGS = {"MiniMaxH3ChannelBalance": MiniMaxH3ChannelBalance}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3ChannelBalance": "MiniMax H3 Channel Balance"}
