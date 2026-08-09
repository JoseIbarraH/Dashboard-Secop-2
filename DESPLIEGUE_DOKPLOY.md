# Desplegar el observatorio en Dokploy

Dokploy corre tus contenedores detrás de Traefik (un reverse proxy). Streamlit es
una app web que escucha en el puerto **8501** y usa **websockets** para la
interactividad; Traefik ya sabe manejar websockets, así que no hay que configurar
nada especial para eso. Lo único crítico es indicarle a Dokploy el **puerto 8501**.

## Qué subir al repositorio de GitHub

En un repo (puede ser el mismo de la app o uno nuevo), pon:

- `app.py`
- `requirements.txt`
- `Dockerfile`
- (opcional) `secop2_limpio.parquet` — solo si en el Dockerfile descomentas la
  línea `COPY secop2_limpio.parquet .`. Si no, la app lo descarga sola desde tu
  repo de datos al arrancar.

## Pasos en el panel de Dokploy

1. **Create Application.** En tu proyecto de Dokploy, crea una nueva *Application*.
2. **Conecta el repositorio.** Elige GitHub como *Source*, selecciona el repo y la
   rama (`main`).
3. **Build Type = Dockerfile.** Dokploy detectará el `Dockerfile` de la raíz. (Si
   está en una subcarpeta, indícale la ruta en *Docker File Path*.)
4. **Deploy.** Lanza el primer build. Verás los logs de construcción.
5. **Dominio y puerto.** Ve a la pestaña **Domains** de la app:
   - Agrega tu dominio o subdominio (o usa el que Dokploy te asigna con Traefik).
   - En **Container Port** pon **8501**. Este paso es el que hace que Traefik
     enrute bien; si lo dejas en otro puerto, verás 502/Bad Gateway.
   - Activa HTTPS (Let's Encrypt) si quieres certificado.
6. **Redeploy** si cambiaste el puerto o el dominio.

Al terminar tendrás una URL pública del observatorio servida por tu Dokploy.

## Comprobaciones si algo falla

- **502 / Bad Gateway:** casi siempre es el *Container Port*. Debe ser **8501**.
- **La página carga pero queda “Please wait…” o no responde a clics:** es el
  websocket. Con Traefik suele funcionar solo; si usas otro proxy delante,
  asegúrate de reenviar los encabezados de *Upgrade/Connection* (websocket).
- **Falla al leer los datos:** si dejaste el `COPY` del parquet comentado, el
  contenedor necesita salida a internet para bajarlo de GitHub la primera vez.
  Si tu servidor no tiene salida, descomenta el `COPY` y sube el `.parquet` al repo.
- **Build lento o pesado:** normal la primera vez (instala pandas, sklearn,
  pyarrow). Los siguientes builds reutilizan caché si no cambia `requirements.txt`.

## Probar la imagen en local (opcional, si tienes Docker)

```bash
docker build -t observatorio .
docker run -p 8501:8501 observatorio
# abre http://localhost:8501
```
