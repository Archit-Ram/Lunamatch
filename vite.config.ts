import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import fs from 'fs';
import { defineConfig, Plugin } from 'vite';

function serveOnnxPlugin(): Plugin {
  return {
    name: 'serve-onnx-models',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url) {
          const urlPath = req.url.split('?')[0];
          if (urlPath.startsWith('/models/') || urlPath.includes('.onnx')) {
            const fileName = path.basename(urlPath);
            const filePath = path.join(process.cwd(), 'public', 'models', fileName);
            if (fs.existsSync(filePath)) {
              const stat = fs.statSync(filePath);
              res.writeHead(200, {
                'Content-Type': 'application/octet-stream',
                'Content-Length': stat.size,
                'Access-Control-Allow-Origin': '*',
                'Cache-Control': 'public, max-age=86400',
              });
              const stream = fs.createReadStream(filePath);
              stream.pipe(res);
              return;
            }
          } else if (urlPath.startsWith('/wasm/')) {
            const fileName = path.basename(urlPath);
            const filePath = path.join(process.cwd(), 'public', 'wasm', fileName);
            if (fs.existsSync(filePath)) {
              const stat = fs.statSync(filePath);
              let contentType = 'application/octet-stream';
              if (fileName.endsWith('.wasm')) {
                contentType = 'application/wasm';
              } else if (fileName.endsWith('.mjs') || fileName.endsWith('.js')) {
                contentType = 'text/javascript';
              }
              res.writeHead(200, {
                'Content-Type': contentType,
                'Content-Length': stat.size,
                'Access-Control-Allow-Origin': '*',
                'Cache-Control': 'public, max-age=86400',
              });
              const stream = fs.createReadStream(filePath);
              stream.pipe(res);
              return;
            }
          }
        }
        next();
      });
    },
  };
}

export default defineConfig(() => {
  return {
    plugins: [react(), tailwindcss(), serveOnnxPlugin()],
    assetsInclude: ['**/*.onnx'],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modifyâfile watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true',
      // Disable file watching when DISABLE_HMR is true to save CPU during agent edits.
      watch: process.env.DISABLE_HMR === 'true' ? null : {},
    },
  };
});
