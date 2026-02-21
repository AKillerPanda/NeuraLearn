import { useState, useEffect, useCallback, memo } from "react";
import { Button } from "./ui/button";
import { Trash2, Save } from "lucide-react";

/* ── Types ────────────────────────────────────────────────────────── */
interface TopicNote {
  text: string;
  updatedAt: string;
}

interface TopicNotesProps {
  topicId: string;
  topicName: string;
  skill: string;
}

/* ── Helpers ──────────────────────────────────────────────────────── */
function storageKey(skill: string): string {
  return `neuralearn_notes_${skill.toLowerCase().replace(/\s+/g, "_")}`;
}

function loadNotes(skill: string): Record<string, TopicNote> {
  try {
    const raw = localStorage.getItem(storageKey(skill));
    if (raw) return JSON.parse(raw);
  } catch { /* ignore */ }
  return {};
}

function saveNotes(skill: string, notes: Record<string, TopicNote>) {
  localStorage.setItem(storageKey(skill), JSON.stringify(notes));
  // Notify same-tab listeners (storage event only fires cross-tab)
  window.dispatchEvent(new CustomEvent("neuralearn-notes-change", { detail: skill }));
}

/* ── Component ────────────────────────────────────────────────────── */
export function TopicNotes({ topicId, topicName, skill }: TopicNotesProps) {
  const [noteText, setNoteText] = useState("");
  const [saved, setSaved] = useState(true);

  // Load note for this topic on mount / topic change
  useEffect(() => {
    const notes = loadNotes(skill);
    const existing = notes[topicId];
    setNoteText(existing?.text ?? "");
    setSaved(true);
  }, [topicId, skill]);

  const handleSave = useCallback(() => {
    const notes = loadNotes(skill);
    if (noteText.trim()) {
      notes[topicId] = {
        text: noteText.trim(),
        updatedAt: new Date().toISOString(),
      };
    } else {
      delete notes[topicId];
    }
    saveNotes(skill, notes);
    setSaved(true);
  }, [noteText, topicId, skill]);

  const handleDelete = useCallback(() => {
    const notes = loadNotes(skill);
    delete notes[topicId];
    saveNotes(skill, notes);
    setNoteText("");
    setSaved(true);
  }, [topicId, skill]);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium flex items-center gap-1.5">
          My Notes — {topicName}
        </h4>
        {!saved && (
          <span className="text-[10px] text-amber-500 font-medium">unsaved</span>
        )}
      </div>
      <textarea
        value={noteText}
        onChange={(e) => {
          setNoteText(e.target.value);
          setSaved(false);
        }}
        placeholder="Jot down your thoughts, key concepts, or questions about this topic..."
        className="w-full h-24 text-sm rounded-lg border border-gray-200 p-2.5 resize-none focus:outline-none focus:ring-2 focus:ring-purple-200 bg-gray-50"
      />
      <div className="flex gap-2">
        <Button
          variant="outline"
          size="sm"
          className="gap-1 text-xs"
          onClick={handleSave}
          disabled={saved}
        >
          <Save className="size-3" />
          Save
        </Button>
        {noteText && (
          <Button
            variant="ghost"
            size="sm"
            className="gap-1 text-xs text-red-500 hover:text-red-600"
            onClick={handleDelete}
          >
            <Trash2 className="size-3" />
            Clear
          </Button>
        )}
      </div>
    </div>
  );
}

/* ── Notes summary (for sidebar) ─────────────────────────────────── */
export const NotesSummary = memo(function NotesSummary({ skill }: { skill: string }) {
  const [notes, setNotes] = useState<Record<string, TopicNote>>({});

  useEffect(() => {
    setNotes(loadNotes(skill));
    const handler = () => setNotes(loadNotes(skill));
    // Cross-tab sync
    window.addEventListener("storage", handler);
    // Same-tab sync via custom event
    window.addEventListener("neuralearn-notes-change", handler);
    return () => {
      window.removeEventListener("storage", handler);
      window.removeEventListener("neuralearn-notes-change", handler);
    };
  }, [skill]);

  const entries = Object.entries(notes);
  if (entries.length === 0) {
    return (
      <p className="text-xs text-gray-400 italic">
        No notes yet. Click a topic to add notes.
      </p>
    );
  }

  return (
    <div className="space-y-2">
      <p className="text-xs text-gray-500">{entries.length} topic(s) with notes</p>
      {entries.slice(0, 5).map(([id, note]) => (
        <div key={id} className="rounded-lg border p-2 bg-gray-50">
          <p className="text-[11px] text-gray-700 line-clamp-2">{note.text}</p>
          <p className="text-[9px] text-gray-400 mt-1">
            {new Date(note.updatedAt).toLocaleDateString()}
          </p>
        </div>
      ))}
      {entries.length > 5 && (
        <p className="text-[10px] text-gray-400">+{entries.length - 5} more</p>
      )}
    </div>
  );
});
