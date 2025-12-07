import { GoogleGenerativeAI } from "@google/generative-ai";
import type { EnvSample } from "./types";

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY || "");

// In-memory cache for rubrics
const rubricCache = new Map<string, string>();

/**
 * Ensure a rubric exists for the given env sample.
 * If rubric is null, generate one using gemini-3-pro-preview.
 */
export async function ensureRubric(envSample: EnvSample): Promise<string> {
  // Return cached rubric if available
  if (envSample.rubric) {
    return envSample.rubric;
  }

  if (rubricCache.has(envSample.id)) {
    return rubricCache.get(envSample.id)!;
  }

  // Generate rubric using teacher model
  const model = genAI.getGenerativeModel({ model: "gemini-3-pro-preview" });

  const prompt = `You are helping evaluate a dialogue assistant. Given the dialogue so far and the ground truth reply, write 2-4 bullet points describing what a good next assistant response should accomplish.

Task: ${envSample.task_description}

Dialogue so far:
${envSample.prompt}

Ground truth response: ${envSample.ground_truth}

Write 2-4 bullet points describing what makes a good response here. Focus on key qualities like relevance, clarity, helpfulness, and appropriateness.`;

  try {
    const result = await model.generateContent(prompt);
    const rubric = result.response.text().trim();
    
    // Cache the rubric
    rubricCache.set(envSample.id, rubric);
    
    return rubric;
  } catch (error) {
    console.error("Error generating rubric:", error);
    // Fallback rubric
    const fallback = "- Be relevant to the user's request\n- Provide helpful information\n- Be clear and concise";
    rubricCache.set(envSample.id, fallback);
    return fallback;
  }
}

/**
 * Score a candidate response using the teacher model.
 * Returns a number between 0 and 1.
 */
export async function scoreResponse({
  envSample,
  candidate,
  rubric,
}: {
  envSample: EnvSample;
  candidate: string;
  rubric: string;
}): Promise<number> {
  const model = genAI.getGenerativeModel({ model: "gemini-3-pro-preview" });

  const prompt = `You are evaluating a dialogue assistant's response.

Task: ${envSample.task_description}

Dialogue context:
${envSample.prompt}

Ground truth response: ${envSample.ground_truth}

Evaluation criteria (rubric):
${rubric}

Candidate response to evaluate: ${candidate}

Rate this candidate response on a scale from 0.0 to 1.0, where:
- 0.0 = completely inappropriate or unhelpful
- 0.5 = somewhat helpful but missing key elements
- 1.0 = excellent, meets all criteria

Respond with ONLY a number between 0.0 and 1.0, nothing else.`;

  try {
    const result = await model.generateContent(prompt);
    const responseText = result.response.text().trim();
    
    // Parse the number, handling various formats
    const match = responseText.match(/([0-1](?:\.\d+)?)/);
    if (match) {
      const score = parseFloat(match[1]);
      // Clamp to [0, 1]
      return Math.max(0, Math.min(1, score));
    }
    
    // If parsing fails, return neutral score
    console.warn("Failed to parse score from:", responseText);
    return 0.5;
  } catch (error) {
    console.error("Error scoring response:", error);
    return 0.5; // Neutral score on error
  }
}
