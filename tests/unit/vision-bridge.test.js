import { describe, it, expect } from "vitest";
import {
  VISION_BRIDGE_MODEL,
  VISION_BRIDGE_TARGET_MODEL,
  VISION_BRIDGE_VISION_MODEL,
  isVisionBridgeModel,
  hasOpenAIVisionInput,
  buildVisionDescriptionBody,
  rewriteBodyWithVisionDescription,
} from "../../open-sse/services/visionBridge.js";

describe("vision bridge helpers", () => {
  const body = {
    model: VISION_BRIDGE_MODEL,
    stream: true,
    messages: [{
      role: "user",
      content: [
        { type: "text", text: "What is wrong in this screenshot?" },
        { type: "image_url", image_url: { url: "data:image/png;base64,abc" } },
      ],
    }],
  };

  it("recognizes the virtual DeepSeek vision bridge model", () => {
    expect(isVisionBridgeModel(VISION_BRIDGE_MODEL)).toBe(true);
    expect(isVisionBridgeModel(`opencode/${VISION_BRIDGE_MODEL}`)).toBe(true);
    expect(isVisionBridgeModel("deepseek-v4-flash-free")).toBe(false);
  });

  it("detects OpenAI image_url inputs", () => {
    expect(hasOpenAIVisionInput(body)).toBe(true);
    expect(hasOpenAIVisionInput({ messages: [{ role: "user", content: "hi" }] })).toBe(false);
  });

  it("builds a non-streaming MiMo vision description request", () => {
    const out = buildVisionDescriptionBody(body);
    expect(out.model).toBe(VISION_BRIDGE_VISION_MODEL);
    expect(out.stream).toBe(false);
    expect(out.messages.at(-1).content[0].text).toContain("Describe the attached image");
  });

  it("rewrites image input into text for the DeepSeek target", () => {
    const out = rewriteBodyWithVisionDescription(body, "The screenshot shows a red error banner.");
    expect(out.model).toBe(VISION_BRIDGE_TARGET_MODEL);
    expect(out.messages[0].role).toBe("system");
    expect(JSON.stringify(out.messages)).not.toContain("image_url");
    expect(JSON.stringify(out.messages)).toContain("red error banner");
  });
});
