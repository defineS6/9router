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
  const openAiBody = {
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

  it("returns false when no image is present", () => {
    expect(hasOpenAIVisionInput({ messages: [{ role: "user", content: "hi" }] })).toBe(false);
  });

  it("detects OpenAI image_url inputs", () => {
    expect(hasOpenAIVisionInput(openAiBody)).toBe(true);
    expect(hasOpenAIVisionInput({ messages: [{ role: "user", content: [{ type: "image_url", image_url: "https://example.test/a.png" }] }] })).toBe(true);
  });

  it("detects and converts OpenAI Responses input_image inputs", () => {
    const body = { input: [{ role: "user", content: [{ type: "input_text", text: "Describe this" }, { type: "input_image", image_url: "https://example.test/a.png" }] }] };
    expect(hasOpenAIVisionInput(body)).toBe(true);
    const out = buildVisionDescriptionBody(body);
    expect(out.messages[0].content[0].text).toContain("Describe this");
    expect(out.messages[0].content[1]).toEqual({ type: "image_url", image_url: { url: "https://example.test/a.png" } });
  });

  it("detects and converts Claude base64 image sources into data URIs", () => {
    const body = { messages: [{ role: "user", content: [{ type: "image", source: { type: "base64", media_type: "image/png", data: "abc" } }] }] };
    expect(hasOpenAIVisionInput(body)).toBe(true);
    expect(buildVisionDescriptionBody(body).messages[0].content[1].image_url.url).toBe("data:image/png;base64,abc");
  });

  it("detects and converts Gemini inlineData and fileData image parts", () => {
    const inlineBody = { contents: [{ parts: [{ text: "What is shown?" }, { inlineData: { mimeType: "image/png", data: "abc" } }] }] };
    const fileBody = { request: { contents: [{ parts: [{ fileData: { mimeType: "image/png", fileUri: "gs://bucket/a.png" } }] }] } };
    expect(hasOpenAIVisionInput(inlineBody)).toBe(true);
    expect(buildVisionDescriptionBody(inlineBody).messages[0].content[1].image_url.url).toBe("data:image/png;base64,abc");
    expect(hasOpenAIVisionInput(fileBody)).toBe(true);
    expect(buildVisionDescriptionBody(fileBody).messages[0].content[1].image_url.url).toBe("gs://bucket/a.png");
  });

  it("builds a non-streaming MiMo request with prompt and images in one user message", () => {
    const out = buildVisionDescriptionBody(openAiBody);
    expect(out.model).toBe(VISION_BRIDGE_VISION_MODEL);
    expect(out.stream).toBe(false);
    expect(out.messages).toHaveLength(1);
    expect(out.messages[0].role).toBe("user");
    expect(out.messages[0].content[0].text).toContain("Describe the attached image");
    expect(out.messages[0].content[0].text).toContain("What is wrong in this screenshot?");
    expect(out.messages[0].content[1].type).toBe("image_url");
  });

  it("rewrites image input into text for DeepSeek and strips all image fields", () => {
    const body = {
      ...openAiBody,
      input: [{ content: [{ type: "input_image", image_url: { url: "x" } }] }],
      contents: [{ parts: [{ inlineData: { mimeType: "image/png", data: "abc" } }] }],
      request: { contents: [{ parts: [{ fileData: { mimeType: "image/png", fileUri: "gs://x" } }] }] },
      messages: [{ role: "user", content: [{ type: "image", source: { url: "https://example.test/a.png" } }, { type: "text", text: "Answer my question" }] }],
    };
    const out = rewriteBodyWithVisionDescription(body, "The screenshot shows a red error banner.");
    const json = JSON.stringify(out);
    expect(out.model).toBe(VISION_BRIDGE_TARGET_MODEL);
    expect(out.messages[0].role).toBe("system");
    expect(json).not.toContain("image_url");
    expect(json).not.toContain("input_image");
    expect(json).not.toContain("inlineData");
    expect(json).not.toContain("fileData");
    expect(json).not.toContain("source");
    expect(json).toContain("red error banner");
    expect(json).toContain("Answer my question");
  });
});
