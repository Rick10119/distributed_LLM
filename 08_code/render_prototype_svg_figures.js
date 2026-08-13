"use strict";

// Render the prototype SVG figures to high-resolution PNG files so that the
// LaTeX report can compile without an external Inkscape installation.

const fs = require("fs");
const path = require("path");
const sharp = require("sharp");

const root = path.resolve(__dirname, "..");
const figureDir = path.join(root, "05_results", "figures");

async function main() {
  const names = fs
    .readdirSync(figureDir)
    .filter((name) => name.endsWith(".svg"))
    .sort();

  for (const name of names) {
    const source = path.join(figureDir, name);
    const target = path.join(figureDir, name.replace(/\.svg$/, ".png"));
    await sharp(source, { density: 240 }).png().toFile(target);
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
