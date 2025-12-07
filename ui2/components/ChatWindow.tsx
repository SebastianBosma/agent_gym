import { ScrollArea } from "@/components/ui/scroll-area";
import { ChatMessage, type ChatMessageProps } from "./ChatMessage";

export interface ChatWindowProps {
  messages: ChatMessageProps[];
}

export function ChatWindow({ messages }: ChatWindowProps) {
  return (
    <ScrollArea className="h-full border rounded-lg">
      <div className="p-4">
        {messages.length === 0 ? (
          <div className="text-center text-gray-500 py-8">
            Click "Run Training" to start the bandit training process...
          </div>
        ) : (
          messages.map((msg, idx) => (
            <ChatMessage key={idx} {...msg} />
          ))
        )}
      </div>
    </ScrollArea>
  );
}
