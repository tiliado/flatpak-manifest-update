from enum import unique, Enum


@unique
class Foreground(str, Enum):
    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BLACK_BOLD = "\033[30;1m"
    RED_BOLD = "\033[31;1m"
    GREEN_BOLD = "\033[32;1m"
    YELLOW_BOLD = "\033[33;1m"
    BLUE_BOLD = "\033[34;1m"
    MAGENTA_BOLD = "\033[35;1m"
    CYAN_BOLD = "\033[36m;1"
    WHITE_BOLD = "\033[37m;1"

    def apply(self, text: str) -> str:
        return f"{self}{text}{self.RESET}"
