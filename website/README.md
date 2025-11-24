# Simple Healthcare Assistant Web UI

This directory contains a minimal HTML/CSS/JS frontend that talks to the Flask chatbot API in `code/app.py`.

## Structure

```
website/
├── index.html   # Page markup + layout
├── styles.css   # Light-weight styling inspired by the original design
├── script.js    # Fetch logic for /api/chat and /api/reset
└── README.md
```

## Usage

1. Start the Flask backend (default: `http://127.0.0.1:5000`).
2. Serve this folder with any static file server, e.g.
   ```bash
   cd website
   python -m http.server 8000
   ```
3. Visit `http://localhost:8000` (or wherever you’re hosting the static files).

The frontend assumes the API is available on the *same origin*. If you host it elsewhere, append `?api=https://your-api-host` to the page URL or set `window.CHATBOT_API_BASE = "https://your-api-host";` before loading `script.js`.

## Customisation

- Update `styles.css` to tweak colours/spacing.
- Edit `index.html` to change copy or layout.
- Extend `script.js` if you want to show additional data returned by the backend.
