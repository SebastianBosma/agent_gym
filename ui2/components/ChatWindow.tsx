import { ScrollArea } from "@/components/ui/scroll-area";
import { ChatMessage, type ChatMessageProps } from "./ChatMessage";

export interface ChatWindowProps {
  messages: ChatMessageProps[];
}

export function ChatWindow({ messages }: ChatWindowProps) {
  // Reverse the messages array to show latest at top
  const reversedMessages = [...messages].reverse();

  return (
    <ScrollArea className="h-full border rounded-lg">
      <div className="p-4 space-y-3">
        {messages.length === 0 ? (
          <div className="text-center text-gray-500 py-8">
            Click "Run Training" to start the bandit training process...
          </div>
        ) : (
          reversedMessages.map((msg, idx) => (
            <ChatMessage key={`msg-${messages.length - 1 - idx}`} {...msg} />
          ))
        )}
      </div>
    </ScrollArea>
  );
}
