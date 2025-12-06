
# 1. Imagen base oficial de Node
FROM node:20-alpine

# 2. Establecer directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiar solo los archivos necesarios para instalar dependencias
COPY package.json package-lock.json ./

# 4. Instalar dependencias SIN usar node_modules local
RUN npm install --production

# 5. Copiar el código restante (excluye node_modules por el .dockerignore)
COPY . .

# 6. Exponer el puerto (cámbialo si tu server usa otro)
EXPOSE 3000

# 7. Comando de inicio
CMD ["node", "server.mjs"]

