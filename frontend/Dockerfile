FROM node:18-alpine

WORKDIR /app

# Install dependencies first for better caching
COPY package*.json ./
RUN npm install

# Copy frontend source code
COPY . .

EXPOSE 5173

# Run Vite dev server exposing host network for dashboard access
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
