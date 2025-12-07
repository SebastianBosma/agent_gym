import { EnvSample } from "./types";
import fs from "fs";
import path from "path";

let cachedEnvData: EnvSample[] | null = null;

/**
 * Load environment data from the JSONL file.
 * Results are cached after first load.
 */
export function loadEnvData(): EnvSample[] {
  if (cachedEnvData) {
    return cachedEnvData;
  }

  const dataPath = path.join(process.cwd(), "data", "env_sgd.jsonl");
  const fileContent = fs.readFileSync(dataPath, "utf-8");
  
  const samples: EnvSample[] = fileContent
    .split("\n")
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line) as EnvSample);

  cachedEnvData = samples;
  return samples;
}

/**
 * Get a random sample from the environment data.
 */
export function getRandomEnvSample(): EnvSample {
  const samples = loadEnvData();
  const randomIndex = Math.floor(Math.random() * samples.length);
  return samples[randomIndex];
}

/**
 * Get a specific sample by ID.
 */
export function getEnvSampleById(id: string): EnvSample | undefined {
  const samples = loadEnvData();
  return samples.find((sample) => sample.id === id);
}
