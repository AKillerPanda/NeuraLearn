# Frontend

React 19 + TypeScript UI built with Vite, ReactFlow for graph visualisation, and shadcn/ui for components. Features a 5-tab sidebar with learning paths, progress tracking, analytics, Pomodoro timer, and gamification.

## Files

| File | What It Does |
|------|-------------|
| **`pages/Home.tsx`** | Landing page. Search bar with live SDS-powered spell correction (debounced 500 ms). Example skill chips for quick start. Submits a skill to generate the knowledge graph. |
| **`pages/KnowledgeGraph.tsx`** | Main view (~1235 lines). Renders the interactive graph via ReactFlow with pan/zoom. **5-tab sidebar**: **Paths** (multiple learning routes with step-by-step timeline, progress bars, stats, "Continue Learning"), **Progress** (node mastery toggles with server validation), **Stats** (spectral analytics, ACO convergence chart, study time estimates, flashcard export), **Timer** (Pomodoro with session estimates), **Rewards** (XP, levels, achievements). Includes dark mode toggle, shortest-path finder, sub-graph deep dives, and node detail dialogs with learning resources. |
| **`components/CustomNode.tsx`** | Custom ReactFlow node component. Displays topic name with colour-coded difficulty badge (foundational/intermediate/advanced/expert), spectral cluster ring, depth glow, prerequisite hints, and resource count. Memoised to avoid unnecessary re-renders. |
| **`components/PomodoroTimer.tsx`** | 25/5/15-minute focus/short-break/long-break timer. Circular SVG progress ring with smooth animation. Auto-switches to break after focus; long break every 4th session. Session counter with total focused time display. Mode-specific colours (purple/green/blue). |
| **`components/GamificationPanel.tsx`** | XP system (100 per topic, 25 per Pomodoro session). Quadratic level scaling with animated XP progress bar. 8 achievements (First Step, Rising Scholar, Halfway There, Completionist, Deep Focus, Study Marathon, On a Roll, Weekly Warrior). Streak tracking via localStorage with date-based calculation. |
| **`components/ui/`** | Full shadcn/ui component library (45+ components) — buttons, cards, tabs, badges, progress bars, dialogs, sheets, tooltips, etc. All styled with Tailwind CSS v4. |
| **`utils/api.ts`** | Typed API client (~360 lines). 12 exported functions covering all 10 backend endpoints plus style helpers. 90-second `AbortController` timeouts. Typed interfaces for `SkillGraphData`, `LearningPath`, `GraphStats`, `MasteryState`, `ShortestPathStep`, `FlashCardExport`, `StudyStats`, and more. `styleNode()` assigns ReactFlow node types and colours based on difficulty level. |
| **`App.tsx`** | Root component with `RouterProvider`. |
| **`routes.tsx`** | Route definitions — `/` (Home) and `/learn/:skill` (graph view). |
| **`main.tsx`** | Entry point. Mounts the React app. |
| **`vite.config.ts`** | Vite config with dev proxy (`/api` → `localhost:5000`) so the frontend can talk to Flask without CORS issues in development. |

## Stack

- **React 19** + **TypeScript 5.7** — type-safe components with strict error typing (`unknown` instead of `any`)
- **Vite 6.4** — fast dev server with HMR
- **ReactFlow** — interactive node-edge graph rendering with pan, zoom, and custom nodes
- **shadcn/ui** — accessible, composable component primitives
- **Tailwind CSS v4** — utility-first styling with dark mode support
- **Motion** — animations (page transitions, card hovers, tab switches)
- **Sonner** — toast notifications with dark/light theme