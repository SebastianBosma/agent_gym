import { cn } from "@/lib/utils";

export type MessageRole = "system" | "env" | "student" | "teacher";

export interface ChatMessageProps {
  role: MessageRole;
  content: string;
  policyName?: string;
}

export function ChatMessage({ role, content, policyName }: ChatMessageProps) {
  const roleColors: Record<MessageRole, string> = {
    system: "bg-blue-50 border-blue-200 text-blue-900",
    env: "bg-amber-50 border-amber-200 text-amber-900",
    student: "bg-green-50 border-green-200 text-green-900",
    teacher: "bg-purple-50 border-purple-200 text-purple-900",
  };

  const roleLabels: Record<MessageRole, string> = {
    system: "System",
    env: "Environment",
    student: policyName ? `Student (${policyName})` : "Student",
    teacher: "Teacher",
  };

  return (
    <div className={cn("rounded-lg border p-4 mb-3", roleColors[role])}>
      <div className="font-semibold mb-1 text-sm">{roleLabels[role]}</div>
      <div className="text-sm whitespace-pre-wrap">{content}</div>
    </div>
  );
}
