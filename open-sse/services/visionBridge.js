export const VISION_BRIDGE_MODEL = "deepseek-v4-flash-free-vision";
export const VISION_BRIDGE_TARGET_MODEL = "opencode/deepseek-v4-flash-free";
export const VISION_BRIDGE_VISION_MODEL = "opencode/mimo-v2.5-free";

const VISION_PROMPT = [
  "Describe the attached image(s) for a downstream text-only reasoning model.",
  "Include visible text, UI elements, layout, objects, errors, and any details relevant to the user's question.",
  "Be factual and concise; do not answer the user's final question unless it is needed to explain the image."
].join(" ");

const BRIDGE_SYSTEM_PROMPT = [
  "You are answering with help from a vision model.",
  "The original image input was converted into a textual description.",
  "Use the description to answer the user's request, and say when the description is insufficient."
].join(" ");

function isImageBlock(block) {
  if (!block || typeof block !== "object") return false;
  if (block.type === "image_url" || block.type === "image" || block.type === "input_image") return true;
  const mime = block.inlineData?.mimeType || block.fileData?.mimeType;
  return typeof mime === "string" && mime.startsWith("image/");
}

export function isVisionBridgeModel(model) {
  return model === VISION_BRIDGE_MODEL || model === `opencode/${VISION_BRIDGE_MODEL}`;
}

export function hasOpenAIVisionInput(body) {
  return Array.isArray(body?.messages) && body.messages.some((message) => (
    Array.isArray(message?.content) && message.content.some(isImageBlock)
  ));
}

export function buildVisionDescriptionBody(body) {
  const messages = Array.isArray(body?.messages) ? body.messages : [];
  return {
    ...body,
    model: VISION_BRIDGE_VISION_MODEL,
    stream: false,
    messages: [
      ...messages,
      { role: "user", content: [{ type: "text", text: VISION_PROMPT }] },
    ],
  };
}

function blockText(block) {
  if (!block || typeof block !== "object") return "";
  if (block.type === "text") return block.text || "";
  if (typeof block.text === "string") return block.text;
  return "";
}

function stripImagesFromContent(content) {
  if (!Array.isArray(content)) return content;
  const kept = content.filter((block) => !isImageBlock(block));
  if (kept.length === 0) return "[Image input was converted to text by the vision bridge.]";
  if (kept.every((block) => block?.type === "text" || typeof block?.text === "string")) {
    return kept.map(blockText).filter(Boolean).join("\n");
  }
  return kept;
}

export function rewriteBodyWithVisionDescription(body, description) {
  const messages = Array.isArray(body?.messages) ? body.messages : [];
  const cleanedMessages = messages.map((message) => ({
    ...message,
    content: stripImagesFromContent(message.content),
  }));

  return {
    ...body,
    model: VISION_BRIDGE_TARGET_MODEL,
    messages: [
      { role: "system", content: BRIDGE_SYSTEM_PROMPT },
      ...cleanedMessages,
      {
        role: "user",
        content: `Vision model description of the original image input:\n\n${description || "[No description returned]"}`,
      },
    ],
  };
}

export async function extractOpenAIText(response) {
  const data = await response.json();
  const content = data?.choices?.[0]?.message?.content ?? data?.choices?.[0]?.delta?.content ?? data?.output_text;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => typeof part === "string" ? part : (part?.text || part?.content || ""))
      .filter(Boolean)
      .join("\n");
  }
  return "";
}
