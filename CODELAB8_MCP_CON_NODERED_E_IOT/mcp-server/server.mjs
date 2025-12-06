import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import express from "express";
import { z } from "zod";

const nodeRedUrl = process.env.NODERED_URL || "http://localhost:1880";

const server = new McpServer({
name: "mcp-nodered-demo",
version: "1.0.0",
});

server.registerTool(
"setLedState",
{
title: "Encender/apagar LED vía Node-RED",
description:
"Envía un comando al endpoint /device/led de Node-RED para controlar un LED",
inputSchema: {
deviceId: z.string(),
state: z.enum(["ON", "OFF"]),
durationSeconds: z.number().optional()
},
outputSchema: {
status: z.string(),
nodeRedResponse: z.any()
}
},
async ({ deviceId, state, durationSeconds }) => {
const payload = {
deviceId,
state,
durationSeconds: durationSeconds ?? 0
};

const res = await fetch(`${nodeRedUrl}/device/led`, {
method: "POST",
headers: { "Content-Type": "application/json" },
body: JSON.stringify(payload)
});

let data;
try {
data = await res.json();
} catch {
data = { raw: await res.text() };
}

const output = {
status: "ok",
nodeRedResponse: data
};

return {
content: [
{
type: "text",
text: JSON.stringify(output)
}
],
structuredContent: output
};
}
);

const app = express();
app.use(express.json());

app.post("/mcp", async (req, res) => {
const transport = new StreamableHTTPServerTransport({
sessionIdGenerator: undefined,
enableJsonResponse: true
});

res.on("close", () => {
transport.close();
});

await server.connect(transport);
await transport.handleRequest(req, res, req.body);
});

const port = parseInt(process.env.PORT || "3000", 10);
app
.listen(port, () => {
console.log(`MCP Server corriendo en http://localhost:${port}/mcp`);
})
.on("error", (error) => {
console.error("Server error:", error);
process.exit(1);
});