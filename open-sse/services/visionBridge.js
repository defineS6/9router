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

function dataUri(mimeType, data) {
  if (!mimeType || !data) return "";
  return `data:${mimeType};base64,${data}`;
}

function isImageBlock(block) {
  if (!block || typeof block !== "object") return false;
  if (block.type === "image_url" || block.type === "image" || block.type === "input_image") return true;
  const mime = block.inlineData?.mimeType || block.fileData?.mimeType || block.source?.media_type;
  return typeof mime === "string" && mime.startsWith("image/");
}

function contentArrays(body) {
  const arrays = [];
  if (Array.isArray(body?.messages)) {
    for (const message of body.messages) if (Array.isArray(message?.content)) arrays.push(message.content);
  }
  if (Array.isArray(body?.input)) {
    for (const item of body.input) if (Array.isArray(item?.content)) arrays.push(item.content);
  }
  const contents = Array.isArray(body?.contents) ? body.contents : body?.request?.contents;
  if (Array.isArray(contents)) {
    for (const item of contents) if (Array.isArray(item?.parts)) arrays.push(item.parts);
  }
  return arrays;
}

function textFromBlock(block) {
  if (!block || typeof block !== "object") return "";
  if (typeof block.text === "string") return block.text;
  if (typeof block.content === "string") return block.content;
  if (block.type === "input_text" && typeof block.text === "string") return block.text;
  return "";
}

function imageBlockToOpenAI(block) {
  if (!isImageBlock(block)) return null;

  if (block.type === "image_url") {
    const url = typeof block.image_url === "string" ? block.image_url : block.image_url?.url;
    return url ? { type: "image_url", image_url: { url, detail: block.image_url?.detail || block.detail || "auto" } } : null;
  }

  if (block.type === "input_image") {
    const url = block.image_url || block.file_id || "";
    return url ? { type: "image_url", image_url: { url, detail: block.detail || "auto" } } : null;
  }

  // Claude image block: { type:"image", source:{ type:"base64", media_type, data } }
  if (block.type === "image" && block.source) {
    const url = block.source.url || dataUri(block.source.media_type, block.source.data);
    return url ? { type: "image_url", image_url: { url, detail: block.detail || "auto" } } : null;
  }

  // Gemini/Antigravity inlineData or fileData part.
  const inlineUrl = dataUri(block.inlineData?.mimeType, block.inlineData?.data);
  const fileUrl = block.fileData?.fileUri || "";
  const url = inlineUrl || fileUrl;
  return url ? { type: "image_url", image_url: { url, detail: block.detail || "auto" } } : null;
}

function collectVisionBridgeInput(body) {
  const textParts = [];
  const imageParts = [];

  for (const content of contentArrays(body)) {
    for (const block of content) {
      const text = textFromBlock(block);
      if (text) textParts.push(text);
      const image = imageBlockToOpenAI(block);
      if (image) imageParts.push(image);
    }
  }

  return { text: textParts.join("\n\n"), images: imageParts };
}

export function isVisionBridgeModel(model) {
  return model === VISION_BRIDGE_MODEL || model === `opencode/${VISION_BRIDGE_MODEL}`;
}

export function hasOpenAIVisionInput(body) {
  return contentArrays(body).some((content) => content.some(isImageBlock));
}

export function buildVisionDescriptionBody(body) {
  const { text, images } = collectVisionBridgeInput(body);
  const prompt = text
    ? `${VISION_PROMPT}\n\nUser request/context:\n${text}`
    : VISION_PROMPT;

  return {
    ...body,
    model: VISION_BRIDGE_VISION_MODEL,
    stream: false,
    messages: [{
      role: "user",
      content: [{ type: "text", text: prompt }, ...images],
    }],
  };
}

function stripImagesFromContent(content) {
  if (!Array.isArray(content)) return content;
  const kept = content.filter((block) => !isImageBlock(block));
  if (kept.length === 0) return "[Image input was converted to text by the vision bridge.]";
  const text = kept.map(textFromBlock).filter(Boolean).join("\n");
  return text || kept;
}

function appendDescriptionToMessages(messages, description) {
  return [
    { role: "system", content: BRIDGE_SYSTEM_PROMPT },
    ...messages.map((message) => ({ ...message, content: stripImagesFromContent(message.content) })),
    {
      role: "user",
      content: `Vision model description of the original image input:\n\n${description || "[No description returned]"}`,
    },
  ];
}

function stripImagesFromInputContent(content) {
  if (!Array.isArray(content)) return content;
  const kept = content.filter((block) => !isImageBlock(block));
  if (kept.length === 0) {
    return [{ type: "input_text", text: "[Image input was converted to text by the vision bridge.]" }];
  }
  return kept;
}

function appendDescriptionToInput(input, description) {
  return [
    ...input.map((item) => ({ ...item, content: stripImagesFromInputContent(item.content) })),
    {
      role: "user",
      content: [{ type: "input_text", text: `Vision model description of the original image input:\n\n${description || "[No description returned]"}` }],
    },
  ];
}

export function rewriteBodyWithVisionDescription(body, description) {
  const rewritten = { ...body, model: VISION_BRIDGE_TARGET_MODEL };

  if (Array.isArray(body?.messages)) {
    rewritten.messages = appendDescriptionToMessages(body.messages, description);
  }

  if (Array.isArray(body?.input)) {
    rewritten.input = appendDescriptionToInput(body.input, description);
  }

  return rewritten;
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
