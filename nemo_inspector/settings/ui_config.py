class UIConfig:
    THEME = {
        "background": "#0D1117",    # GitHub dark background
        "surface": "#161B22",       # GitHub dark surface
        "primary": "#58A6FF",       # GitHub blue
        "primary_hover": "#1F6FEB", # GitHub blue hover
        "text": "#C9D1D9",         # GitHub primary text
        "text_secondary": "#8B949E", # GitHub secondary text
        "border": "#30363D",        # GitHub border color
        "success": "#238636",       # GitHub success green
        "warning": "#9E6A03",       # GitHub warning yellow
        "error": "#F85149",         # GitHub error red
    }
    
    TYPOGRAPHY = {
        "font_weight_normal": "500",    # Medium instead of normal
        "font_weight_bold": "600",      # Semi-bold
        "font_weight_header": "600",    # Semi-bold for headers
        "font_size_normal": "0.9375rem", # Slightly larger than default
        "font_family": "'SF Pro Text', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif"
    }

    STYLES = {
        "text_default": {
            "color": THEME["text"],
            "font-weight": TYPOGRAPHY["font_weight_normal"],
            "font-family": TYPOGRAPHY["font_family"],
            "font-size": TYPOGRAPHY["font_size_normal"]
        },
        "modal_title": {
            "color": THEME["text"],
            "font-weight": TYPOGRAPHY["font_weight_header"],
            "font-size": "1.125rem"
        },
        "pre_text": {
            "color": THEME["text"],
            "font-weight": TYPOGRAPHY["font_weight_normal"],
            "font-family": "monospace",
            "font-size": TYPOGRAPHY["font_size_normal"]
        },
        "card": {
            "background-color": THEME["surface"],
            "border": f"1px solid {THEME['border']}",
            "border-radius": "6px",
            "padding": "1rem",
            "margin-bottom": "1rem",
            "transition": "all 0.2s ease-in-out",
        },
        "input": {
            "background-color": THEME["background"],
            "color": THEME["text"],
            "border": f"1px solid {THEME['border']}",
            "border-radius": "6px",
            "padding": "0.5rem",
        },
        # Add other styles as needed
        "button_primary": {
            "font-weight": 600,
            "background-color": THEME["primary"],
            "color": "white",
            "border": "none",
            "border-radius": "6px",
            "padding": "0.5rem 1rem",
            "font-size": "0.875rem",
            "transition": "all 0.2s ease-in-out",
            # "hover": {
            #     "background-color": THEME["primary_hover"],
            #     "transform": "translateY(-1px)"
            # }
        },
        "button_secondary": {
            "font-weight": 600,
            # "background-color": "transparent",
            "color": THEME["text"],
            "border": f"2px solid {THEME['border']}",
            "border-radius": "6px",
            "padding": "0.5rem 1rem",
            "font-size": "0.875rem",
            "transition": "all 0.2s ease-in-out",
            # "hover": {
            #     "background-color": THEME["surface"],
            #     "border-color": THEME["primary"]
            # }
        },
        "modal": {
            "background-color": THEME["surface"],
            "color": THEME["text"],
            "border-radius": "8px",
            "border": f"1px solid {THEME['border']}",
            "box-shadow": "0 8px 24px rgba(0,0,0,0.2)",
            "width": "100%",
            "max-width": "800px",
            "margin": "0 auto",
        },
        "button_danger": {
            "background-color": THEME["error"],
            "color": "white",
            "border": "none",
            "border-radius": "6px",
            "padding": "0.5rem 1rem",
            "font-size": "0.875rem",
            "transition": "all 0.2s ease-in-out",
            "hover": {
                "background-color": "#DA3633",
                "transform": "translateY(-1px)"
            }
        },
        "input_group": {
            "background-color": THEME["background"],
            "border": f"1px solid {THEME['border']}",
            "border-radius": "6px",
            "overflow": "hidden"
        },
        "input": {
            "background-color": THEME["background"],
            "color": THEME["text"],
            "border": f"1px solid {THEME['border']}",
            "border-radius": "6px",
            "padding": "0.5rem",
            "font-size": "0.875rem"
        },
        "icon_button": {
            "background-color": "transparent",
            "border": "none",
            "padding": "4px",
            "display": "inline-flex",
            "align-items": "center",
            "justify-content": "center",
            "border-radius": "4px",
            "transition": "all 0.2s ease-in-out",
            "hover": {
                "background-color": "rgba(255, 255, 255, 0.1)",
            }
        },
        "switch": {
            "color": THEME["text"],
            "background-color": THEME["surface"],
            "border": f"1px solid {THEME['border']}",
            "border-radius": "6px",
            "padding": "0.5rem",
            "margin": "0.5rem 0",
            "transition": "all 0.2s ease-in-out",
        },
        "modal_section": {
            "border-bottom": f"1px solid {THEME['border']}",
            "padding": "1rem",
            "background-color": THEME["surface"],
        },

    }