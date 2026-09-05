# SatQueryAI Frontend

The frontend application for **SatQueryAI**, an interactive vision-language assistant for multimodal remote sensing image analysis.

This web interface allows users to upload remote sensing images, enter natural-language queries, and view the analysis results returned by the SatQueryAI backend.

## Tech Stack

* React
* Vite
* JavaScript
* Tailwind CSS

## Features

* Interactive satellite image analysis interface
* Remote sensing image upload
* Natural-language query input
* Support for optical and SAR imagery
* Query processing through the SatQueryAI backend
* Routing and analysis status visualization
* AI-generated analysis results
* Demo mode for testing the interface

## Project Structure

```text
frontend/
├── public/
├── src/
│   ├── api/
│   │   └── satqueryApi.js
│   │
│   ├── components/
│   │   ├── BossNode.jsx
│   │   ├── DemoModeToggle.jsx
│   │   ├── ErrorBanner.jsx
│   │   ├── ImageThumbnail.jsx
│   │   ├── QueryInput.jsx
│   │   └── ResultPanel.jsx
│   │
│   ├── demo/
│   ├── hooks/
│   │   └── useSatQuery.js
│   │
│   └── lib/
│
├── index.html
├── package.json
├── package-lock.json
└── vite.config.js
```

## Requirements

Make sure the following are installed:

* Node.js
* npm

You can verify your installation with:

```bash
node --version
npm --version
```

## Installation

From the project root:

```bash
cd frontend
```

Install the dependencies:

```bash
npm install
```

## Run the Frontend

Start the Vite development server:

```bash
npm run dev
```

The frontend will normally be available at:

```text
http://localhost:5173
```

Open this URL in your browser.

## Backend Connection

The frontend communicates with the SatQueryAI backend through the API layer located in:

```text
src/api/satqueryApi.js
```

The backend should be running separately when testing features that require real API communication.

## Production Build

To create a production build:

```bash
npm run build
```

The build output is generated in:

```text
dist/
```

To preview the production build locally:

```bash
npm run preview
```

## Development

For development, make changes inside the `src/` directory and restart the development server only if required.

The main UI components are located in:

```text
src/components/
```

API communication is handled through:

```text
src/api/
```

and application-level query handling is managed through:

```text
src/hooks/
```

## Notes

The `node_modules/` and `dist/` directories are generated locally and should not be committed to Git.

The frontend is designed to work as the user-facing interface of the larger SatQueryAI system.
