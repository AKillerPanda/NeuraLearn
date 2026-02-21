# Frontend

React 19 + TypeScript UI built with Vite, ReactFlow for graph visualisation, and shadcn/ui for components. Originally designed in Figma, then implemented and refined.

## Files

| File | What It Does |
|------|-------------|
| **`pages/Home.tsx`** | Landing page. Search bar with live SDS-powered spell correction (debounced). Submits a skill to generate the knowledge graph. |
| **`pages/KnowledgeGraph.tsx`** | Main view. Renders the interactive graph via ReactFlow with pan/zoom. Sidebar has two tabs — **Topics** (node list with level badges and mastery toggles) and **Paths** (multiple learning routes with step-by-step timeline, progress bars, stats, and a "Continue Learning" button). |
| **`components/CustomNode.tsx`** | Custom ReactFlow node component. Displays topic name with colour-coded difficulty badge (foundational/intermediate/advanced/expert). Memoised to avoid unnecessary re-renders. |
| **`components/figma/ImageWithFallback.tsx`** | Image component with graceful fallback — carried over from the Figma design system. |
| **`components/ui/`** | Full shadcn/ui component library (45+ components) — buttons, cards, tabs, badges, progress bars, dialogs, tooltips, etc. All styled with Tailwind CSS v4. |
| **`utils/api.ts`** | Typed API client. Handles `POST /api/generate`, `POST /api/spell-check`, and `POST /api/sub-graph` with 90-second `AbortController` timeouts. Includes `styleNode()` which assigns ReactFlow node types and colours based on difficulty level. |
| **`App.tsx`** | Root component with route provider. |
| **`routes.tsx`** | Route definitions — `/` (Home) and `/knowledge-graph` (graph view). |
| **`main.tsx`** | Entry point. Mounts the React app. |
| **`vite.config.ts`** | Vite config with dev proxy (`/api` → `localhost:5000`) so the frontend can talk to Flask without CORS issues in development. |

## Stack

- **React 19** + **TypeScript 5.7** — type-safe components
- **Vite 6.4** — fast dev server with HMR
- **ReactFlow** — interactive node-edge graph rendering with pan, zoom, and custom nodes
- **shadcn/ui** — accessible, composable component primitives
- **Tailwind CSS v4** — utility-first styling
- **Motion** — animations
- **Sonner** — toast notifications