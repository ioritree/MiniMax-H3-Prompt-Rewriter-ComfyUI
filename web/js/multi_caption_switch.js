import { app } from "../../scripts/app.js";


// Both nodes that grow the four reference groups: the caption-only one and
// the one that captions and writes in the same run. They share the input
// prefixes and the mask widget, so they share these switches too.
const NODES = new Set(["MiniMaxH3MultiReferenceCaption", "MiniMaxH3ReferenceWriterRef"]);
const MASK = "enabled_mask";
const PREFIXES = ["subject_", "picture_", "video_", "audio_"];

const BOX = 14;
const LABEL_X = 24;
const GAP = 14;
const RIGHT_INSET = 22;
const COLUMN = "__minimaxH3SwitchColumn";
const ON_FILL = "#4A9D5B";
const OFF_FILL = "#2B2B2B";
const BORDER = "#6A6A6A";
const TICK = "#FFFFFF";
const DIMMED = 0.35;

function slotName(input) {
    const name = input?.name;
    if (!name) return null;
    const tail = name.slice(name.lastIndexOf(".") + 1);
    return PREFIXES.some((prefix) => tail.startsWith(prefix)) ? tail : null;
}

function slotIndices(node) {
    const found = [];
    const inputs = node.inputs || [];
    for (let i = 0; i < inputs.length; i++) {
        if (slotName(inputs[i])) found.push(i);
    }
    return found;
}

function isConnected(node, index) {
    const link = node.inputs?.[index]?.link;
    return link !== null && link !== undefined;
}

function maskWidget(node) {
    return node.widgets?.find((w) => w.name === MASK);
}

function readMask(node) {
    try {
        const parsed = JSON.parse(maskWidget(node)?.value ?? "{}");
        return parsed && typeof parsed === "object" ? parsed : {};
    } catch (error) {
        return {};
    }
}

function isEnabled(mask, name) {
    return mask[name] === undefined ? true : !!mask[name];
}

function toggle(node, name) {
    const mask = readMask(node);
    mask[name] = !isEnabled(mask, name);
    for (const key of Object.keys(mask)) {
        if (mask[key] !== false) delete mask[key];
    }
    const widget = maskWidget(node);
    if (widget) widget.value = JSON.stringify(mask);
}

function hide(widget) {
    widget.hidden = true;
    widget.computeSize = () => [0, -4];
}

function drawBox(ctx, cx, cy, checked, connected) {
    const x = cx - BOX / 2;
    const y = cy - BOX / 2;
    const r = 3;

    ctx.save();
    ctx.globalAlpha = connected ? 1.0 : DIMMED;

    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + BOX, y, x + BOX, y + BOX, r);
    ctx.arcTo(x + BOX, y + BOX, x, y + BOX, r);
    ctx.arcTo(x, y + BOX, x, y, r);
    ctx.arcTo(x, y, x + BOX, y, r);
    ctx.closePath();
    ctx.fillStyle = checked ? ON_FILL : OFF_FILL;
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = BORDER;
    ctx.stroke();

    if (checked) {
        ctx.strokeStyle = TICK;
        ctx.lineWidth = 2;
        ctx.lineJoin = "round";
        ctx.beginPath();
        ctx.moveTo(x + BOX * 0.24, y + BOX * 0.52);
        ctx.lineTo(x + BOX * 0.43, y + BOX * 0.72);
        ctx.lineTo(x + BOX * 0.78, y + BOX * 0.28);
        ctx.stroke();
    }

    ctx.restore();
}

function rowCentre(node, index, scratch) {
    node.getConnectionPos(true, index, scratch);
    return scratch[1] - node.pos[1];
}

function columnX(node, ctx) {
    if (!ctx) return node[COLUMN] ?? LABEL_X + 70 + GAP;
    ctx.save();
    ctx.font = `${window.LiteGraph?.NODE_TEXT_SIZE ?? 14}px Arial`;
    let widest = 0;
    for (const index of slotIndices(node)) {
        widest = Math.max(widest, ctx.measureText(slotName(node.inputs[index])).width);
    }
    ctx.restore();
    node[COLUMN] = Math.min(LABEL_X + widest + GAP, node.size[0] - RIGHT_INSET);
    return node[COLUMN];
}

function addSwitches(nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const result = onNodeCreated?.apply(this, arguments);
        const widget = maskWidget(this);
        if (widget) hide(widget);
        return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
        const result = onConfigure?.apply(this, arguments);
        const widget = maskWidget(this);
        if (widget) hide(widget);
        return result;
    };

    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
        onDrawForeground?.apply(this, arguments);
        if (this.flags?.collapsed) return;

        const mask = readMask(this);
        const cx = columnX(this, ctx);
        const scratch = new Float32Array(2);
        for (const index of slotIndices(this)) {
            const name = slotName(this.inputs[index]);
            drawBox(ctx, cx, rowCentre(this, index, scratch), isEnabled(mask, name), isConnected(this, index));
        }
    };

    const onMouseDown = nodeType.prototype.onMouseDown;
    nodeType.prototype.onMouseDown = function (event, pos) {
        if (!this.flags?.collapsed) {
            const cx = columnX(this, null);
            const reach = BOX / 2 + 3;
            const scratch = new Float32Array(2);
            for (const index of slotIndices(this)) {
                const cy = rowCentre(this, index, scratch);
                if (Math.abs(pos[0] - cx) <= reach && Math.abs(pos[1] - cy) <= reach) {
                    toggle(this, slotName(this.inputs[index]));
                    this.setDirtyCanvas(true, true);
                    return true;
                }
            }
        }
        return onMouseDown ? onMouseDown.apply(this, arguments) : undefined;
    };
}

app.registerExtension({
    name: "minimax_h3_rewriter.multi_caption_switch",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (NODES.has(nodeData.name)) addSwitches(nodeType);
    },
});
