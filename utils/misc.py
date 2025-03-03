from IPython.display import display, HTML
import torch

__version__ = "1.0.1"

def get_torch_optimal_device(verbose: bool = True) -> str:
    """
    """
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using {device} device")

    return device


def print_color(text: str, values: list[float], text_color: str = "white", bck_color: int = 0, font_size: int|str = "18px") -> None:
    """
    Display a background-colored text regardind values (of intensity) of each char.

    Args:
        - text (str): string to be displayed char by char
        - values (list[float]): values (one by chars of provided text) in [0.; 1.]
        - text_color (str): CSS name of a color
        - bck_color (int): 0=red, 1=green, 3=blue
        - font_size (int|str): CSS font size
    """
    assert len(text) == len(values), f"lengths of text ({len(text)}) and values ({len(values)}) must be equal."
    assert bck_color in [0, 1, 2], f"bck_color ({bck_color}) must be in [0, 1, 2]"

    def shade_of_value(value, c):
        rgb_color = [
            lambda i : f'rgb({i}, 0, 0)',
            lambda i : f'rgb(0, {i}, 0)',
            lambda i : f'rgb(0, 0, {i})'
        ]
        intensity = int(value * 255)
        return rgb_color[c](intensity)

    def html_span(text_color, background_color, char):
        return f'<span style="font-size:{font_size}; color:\'{text_color}\'; background-color:{background_color}">{char}</span>'

    html_text = [html_span(text_color, shade_of_value(values[i], bck_color), char) for i, char in enumerate(text)]
    all_html_text = ''.join(html_text)
    
    display(HTML(all_html_text))


def update_top_k(top_values, top_indices, new_values, new_indices, k=20):
    """
    """
    combined_values = torch.cat([top_values, new_values])
    combined_indices = torch.cat([top_indices, new_indices])
    
    new_top_values, topk_indices = torch.topk(combined_values, k)
    new_top_indices = combined_indices[topk_indices]
    
    return new_top_values, new_top_indices

def clean_memory(device):
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()