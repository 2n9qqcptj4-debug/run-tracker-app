import streamlit as st

def inject_css():
    theme = st.session_state.get("theme", "dark")
    compact = st.session_state.get("compact_mode", False)

    if theme == "light":
        bg = "#F7F7FA"
        card_bg = "#FFFFFF"
        text = "#111827"
        border = "rgba(0,0,0,0.06)"
    else:
        bg = "#05060A"
        card_bg = "#111827"
        text = "#F9FAFB"
        border = "rgba(255,255,255,0.08)"

    card_padding = "0.8rem 1.0rem" if compact else "1.1rem 1.4rem"
    card_margin = "0.6rem" if compact else "1.0rem"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            background-color: {bg};
            color: {text};
        }}

        .block-container {{
            padding-top: 1.0rem;
            padding-bottom: 4rem;
            max-width: 1200px;
        }}

        .card {{
            background: {card_bg};
            padding: {card_padding};
            border-radius: 14px;
            margin-bottom: {card_margin};
            border: 1px solid {border};
            box-shadow: 0 12px 28px rgba(0,0,0,0.25);
            transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        }}
        .card:hover {{
            transform: translateY(-1px);
            box-shadow: 0 18px 40px rgba(0,0,0,0.32);
            border-color: rgba(59,130,246,0.6);
        }}

        .feed-card {{
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }}

        .feed-header-line {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.35rem;
        }}

        .feed-main-metrics {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.8rem;
            align-items: baseline;
        }}

        .big-distance {{
            font-size: 1.6rem;
            font-weight: 600;
        }}

        .muted {{
            font-size: 0.85rem;
            opacity: 0.7;
        }}

        .tag {{
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 500;
            color: white;
        }}
        .tag-Easy {{ background: #22C55E; }}
        .tag-LongRun {{ background: #3B82F6; }}
        .tag-Tempo {{ background: #F97316; }}
        .tag-Intervals {{ background: #EC4899; }}
        .tag-Recovery {{ background: #64748B; }}
        .tag-Race {{ background: #A855F7; }}
        .tag-Other {{ background: #6B7280; }}

        .pr-banner {{
            background: linear-gradient(90deg, #ffb300, #ffdd66);
            padding: 12px 16px;
            margin: 12px 0;
            border-radius: 10px;
            font-weight: 600;
            font-size: 1rem;
            color: #111827;
            text-align: center;
        }}

        .pr-mini {{
            background: #ffe29a;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-left: 8px;
            display: inline-block;
        }}

        .pr-badge {{
            background: #ffcc00;
            padding: 4px 10px;
            border-radius: 10px;
            display: inline-block;
            margin-top: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            color: #111827;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
