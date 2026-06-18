export const VISION_BRIDGE_MODEL = "deepseek-v4-flash-free-vision";
export const VISION_BRIDGE_TARGET_MODEL = "opencode/deepseek-v4-flash-free";
export const VISION_BRIDGE_VISION_MODEL = "opencode/mimo-v2.5-free";

const VISION_PROMPT = [
  "Describe the attached image(s) for a downstream text-only reasoning model.",
  "Include visible text, UI elements, layout, objects, errors, and any details relevant to the user's question.",
  "Use the user's text below as context, but do not answer the final question unless needed to explain the image.",
].join(" ");

const BRIDGE_SYSTEM_PROMPT = [
  "You are answering with help from a vision model.",
  "The original image input was converted into a textual description; that description may be incomplete or imperfect.",
  "Use the description and the user's original text to answer, and say when the description is insufficient.",
].join(" ");

function dataUri(mimeType, data) {
  return `data:${mimeType || "image/png"};base64,${data}`;
}

function imageUrlValue(value) {
  if (typeof value === "string") return value;
  if (value && typeof value === "object" && typeof value.url === "string") return value.url;
  return null;
}

function toOpenAIImagePart(block) {
  if (!block || typeof block !== "object") return null;

  if (block.type === "image_url") {
    const url = imageUrlValue(block.image_url);
    return url ? { type: "image_url", image_url: { url } } : null;
  }

  if (block.type === "input_image") {
    const url = imageUrlValue(block.image_url || block.url);
    return url ? { type: "image_url", image_url: { url } } : null;
  }

  if (block.type === "image" && block.source) {
    const source = block.source;
    if (typeof source.url === "string") return { type: "image_url", image_url: { url: source.url } };
    if (source.type === "base64" && typeof source.data === "string") {
      return { type: "image_url", image_url: { url: dataUri(source.media_type, source.data) } };
    }
  }

  if (block.inlineData && typeof block.inlineData.data === "string") {
    return { type: "image_url", image_url: { url: dataUri(block.inlineData.mimeType, block.inlineData.data) } };
  }

  if (block.fileData && typeof block.fileData.fileUri === "string") {
    return { type: "image_url", image_url: { url: block.fileData.fileUri } };
  }

  return null;
}

function isImageBlock(block) {
  return Boolean(toOpenAIImagePart(block));
}

function contentBlocks(content) {
  if (Array.isArray(content)) return content;
  if (typeof content === "string") return [{ type: "text", text: content }];
  if (content && typeof content === "object") return [content];
  return [];
}

function getContentArrays(body) {
  const arrays = [];
  for (const message of body?.messages || []) arrays.push(contentBlocks(message?.content));
  for (const item of body?.input || []) arrays.push(contentBlocks(item?.content));
  for (const content of body?.contents || []) arrays.push(contentBlocks(content?.parts));
  for (const content of body?.request?.contents || []) arrays.push(contentBlocks(content?.parts));
  return arrays;
}

function extractTextFromBlock(block) {
  if (typeof block === "string") return block;
  if (!block || typeof block !== "object" || isImageBlock(block)) return "";
  if (typeof block.text === "string") return block.text;
  if (typeof block.content === "string") return block.content;
  if (typeof block.input_text === "string") return block.input_text;
  return "";
}

function collectUserText(body) {
  const chunks = [];
  for (const message of body?.messages || []) {
    const text = contentBlocks(message?.content).map(extractTextFromBlock).filter(Boolean).join("\n");
    if (text) chunks.push(text);
  }
  for (const item of body?.input || []) {
    const text = contentBlocks(item?.content).map(extractTextFromBlock).filter(Boolean).join("\n");
    if (text) chunks.push(text);
  }
  for (const content of body?.contents || []) {
    const text = contentBlocks(content?.parts).map(extractTextFromBlock).filter(Boolean).join("\n");
    if (text) chunks.push(text);
  }
  for (const content of body?.request?.contents || []) {
    const text = contentBlocks(content?.parts).map(extractTextFromBlock).filter(Boolean).join("\n");
    if (text) chunks.push(text);
  }
  return chunks.join("\n\n");
}

function collectImageParts(body) {
  return getContentArrays(body).flatMap((blocks) => blocks.map(toOpenAIImagePart).filter(Boolean));
}

export function isVisionBridgeModel(model) {
  return model === VISION_BRIDGE_MODEL || model === `opencode/${VISION_BRIDGE_MODEL}`;
}

export function hasOpenAIVisionInput(body) {
  return collectImageParts(body).length > 0;
}

export function buildVisionDescriptionBody(body) {
  const userText = collectUserText(body);
  const text = userText ? `${VISION_PROMPT}\n\nUser text/context:\n${userText}` : VISION_PROMPT;
  return {
    model: VISION_BRIDGE_VISION_MODEL,
    stream: false,
    messages: [{ role: "user", content: [{ type: "text", text }, ...collectImageParts(body)] }],
  };
}

function stripImages(value) {
  if (Array.isArray(value)) {
    return value.map(stripImages).filter((item) => item !== undefined);
  }
  if (!value || typeof value !== "object") return value;
  if (isImageBlock(value)) return undefined;
  const next = {};
  for (const [key, child] of Object.entries(value)) {
    if (["image_url", "inlineData", "fileData", "source"].includes(key)) continue;
    const stripped = stripImages(child);
    if (stripped !== undefined) next[key] = stripped;
  }
  return next;
}

export function rewriteBodyWithVisionDescription(body, description) {
  const cleaned = stripImages(body) || {};
  const originalText = collectUserText(body);
  const messages = Array.isArray(cleaned.messages) ? cleaned.messages : [];
  return {
    ...cleaned,
    model: VISION_BRIDGE_TARGET_MODEL,
    messages: [
      { role: "system", content: BRIDGE_SYSTEM_PROMPT },
      ...messages,
      {
        role: "user",
        content: [
          `Vision model description of the original image input:\n\n${description || "[No description returned]"}`,
          originalText ? `Original user text/context:\n\n${originalText}` : null,
        ].filter(Boolean).join("\n\n"),
      },
    ],
  };
}

export async function extractOpenAIText(response) {
  const data = await response.json();
  const content = data?.choices?.[0]?.message?.content ?? data?.choices?.[0]?.delta?.content ?? data?.output_text;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content.map((part) => typeof part === "string" ? part : (part?.text || part?.content || "")).filter(Boolean).join("\n");
  }
  return "";
}
