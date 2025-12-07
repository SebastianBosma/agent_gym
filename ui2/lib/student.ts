import { GoogleGenerativeAI } from "@google/generative-ai";
import type { EnvSample, Policy } from "./types";

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY || "");

/**
 * Generate a student response using gemini-2.0-flash-lite.
 */
export async function generateStudentResponse({
  policy,
  envSample,
}: {
  policy: Policy;
  envSample: EnvSample;
}): Promise<string> {
  const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash-lite" });

  // Construct the chat-style prompt
  const systemInstruction = `${policy.systemPrompt}

Task context: ${envSample.task_description}`;

  const userMessage = envSample.prompt;

  try {
    const chat = model.startChat({
      history: [
        {
          role: "user",
          parts: [{ text: systemInstruction }],
        },
        {
          role: "model",
          parts: [{ text: "I understand. I'll follow these instructions." }],
        },
      ],
    });

    const result = await chat.sendMessage(userMessage);
    return result.response.text().trim();
  } catch (error) {
    console.error("Error generating student response:", error);
    return "I apologize, but I'm having trouble generating a response.";
  }
}
