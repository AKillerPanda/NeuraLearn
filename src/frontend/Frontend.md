# Frontend

React 19 + TypeScript UI built with Vite, ReactFlow for graph visualisation, and shadcn/ui for components. Features a 5-tab sidebar with learning paths, progress tracking, analytics, Pomodoro timer, and gamification.

## Files

| File | What It Does |
| ------ | ------------- |
| **`pages/Home.tsx`** | Landing page. Search bar with live SDS-powered spell correction (debounced 500 ms). Example skill chips for quick start. Submits a skill to generate the knowledge graph. |
| **`pages/KnowledgeGraph.tsx`** | Main view (~1318 lines). Renders the interactive graph via ReactFlow with pan/zoom. **5-tab sidebar**: **Paths** (multiple learning routes with step-by-step timeline, progress bars, stats, "Continue Learning"), **Progress** (node mastery toggles with server validation), **Stats** (spectral analytics, ACO convergence chart, study time estimates, flashcard export, learning insights), **Timer** (Pomodoro with session estimates, weekly study goal), **Rewards** (XP, levels, achievements, study streak calendar). Includes dark mode toggle, shortest-path finder, sub-graph deep dives, topic notes, smart recommendations, and node detail dialogs with learning resources. |
| **`components/CustomNode.tsx`** | Custom ReactFlow node component. Displays topic name with colour-coded difficulty badge (foundational/intermediate/advanced/expert), spectral cluster ring, depth glow, prerequisite hints, and resource count. Memoised to avoid unnecessary re-renders. |
| **`components/PomodoroTimer.tsx`** | 25/5/15-minute focus/short-break/long-break timer. Circular SVG progress ring with smooth animation. Auto-switches to break after focus; long break every 4th session. Session counter with total focused time display. Mode-specific colours (purple/green/blue). |
| **`components/GamificationPanel.tsx`** | XP system (100 per topic, 25 per Pomodoro session). Quadratic level scaling with animated XP progress bar. 8 achievements (First Step, Rising Scholar, Halfway There, Completionist, Deep Focus, Study Marathon, On a Roll, Weekly Warrior). Streak tracking via localStorage with date-based calculation. |
| **`components/LearningInsightsPanel.tsx`** | Displays colour-coded cards for four learning insight dimensions: curriculum cohesion, bottleneck risk, prerequisite load, and curriculum shape. Summarises graph health at a glance. |
| **`components/SmartRecommendation.tsx`** | GAT-based personalised "next topic" recommendations. Shows difficulty bars (Easy/Moderate/Challenging/Hard) with an `onSelect` callback to jump to the recommended node. |
| **`components/StudyStreakCalendar.tsx`** | GitHub-style heatmap calendar tracking daily study streaks. Computes current/longest streaks, persists data to `localStorage`. Exports `recordStudyDay()` helper. |
| **`components/TopicNotes.tsx`** | Per-topic rich-text note editor with save/delete. Persists notes to `localStorage` per skill. Exports both `TopicNotes` and `NotesSummary` components. |
| **`components/WeeklyStudyGoal.tsx`** | Weekly study goal tracker with adjustable target (hours). Auto-resets each Monday, progress bar, persisted to `localStorage`. |
| **`components/ui/`** | Full shadcn/ui component library (45+ components) — buttons, cards, tabs, badges, progress bars, dialogs, sheets, tooltips, etc. All styled with Tailwind CSS v4. |
| **`utils/api.ts`** | Typed API client (~436 lines). 10 exported functions covering all 11 backend endpoints. 90-second `AbortController` timeouts. 18 typed interfaces including `SkillGraphData`, `LearningPath`, `GraphStats`, `MasteryState`, `ShortestPathStep`, `FlashCardExport`, `StudyStats`, `DifficultyScores`, `DifficultyRecommendation`, `LearningInsights`, and more. `styleNode()` assigns ReactFlow node types and colours based on difficulty level. |
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
