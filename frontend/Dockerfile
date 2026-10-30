# fair-football-fee UI: Next.js standalone server. Pages are static shells; the browser fetches all
# data from the API itself, so this container never talks to the API.
FROM node:24-trixie-slim AS build
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# Next.js inlines NEXT_PUBLIC_* into the browser bundle at build time, so the API address the browser
# will use is a build argument; changing it means rebuilding the image
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/v1
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL
RUN npm run build

FROM node:24-trixie-slim
WORKDIR /app
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 HOSTNAME=0.0.0.0 PORT=3000
# server.js serves .next/static itself; the node user owns .next so the server can write its cache
COPY --from=build --chown=node:node /app/.next/standalone ./
COPY --from=build --chown=node:node /app/.next/static ./.next/static
USER node
EXPOSE 3000
CMD ["node", "server.js"]
