# Desplegar el observatorio en Dokploy

Dokploy corre tus contenedores detrás de Traefik (un reverse proxy). Streamlit es
una app web que escucha en el puerto **8501** y usa **websockets** para la
interactividad; Traefik ya sabe manejar websockets, así que no hay que configurar
nada especial para eso. Lo único crítico es indicarle a Dokploy el **puerto 8501**.

## Qué subir al repositorio de GitHub

En un repo (puede ser el mismo de la app o uno nuevo), pon:

- `app.py`
- `pipeline.py`
- `lecturas.py`
- `requirements.txt`
- `.streamlit/config.toml` (el tema; sin él la app se ve mal en modo oscuro)
- `Dockerfile`
- (opcional) `secop2_limpio.parquet` — solo si en el Dockerfile descomentas la
  línea `COPY secop2_limpio.parquet .`. Si no, la app lo descarga sola desde el
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

## Variables de entorno (opcionales)

En la pestaña **Environment** de la aplicación:

| Variable | Por defecto | Para qué |
|---|---|---|
| `SECOP_DATA_REF` | `main` | Rama o SHA de commit del repositorio de datos. Fíjalo a un SHA si necesitas que los números no cambien entre despliegues. |
| `SECOP_DATA_URL` | — | URL completa alternativa al parquet (por ejemplo, un bucket propio). Tiene prioridad. |
| `SECOP_DATA_TIMEOUT` | `60` | Segundos máximos de espera en la descarga. Súbelo si tu salida a internet es lenta. |

## Si el observatorio va a ser público

Está pensado para eso, pero conviene saber dónde están los límites.

**Memoria.** La app carga solo las 12 columnas que usa (de las 85 del dataset),
así que los datos ocupan unos **51 MB** en vez de 195 MB. Con el intérprete y
Streamlit, el contenedor se mueve alrededor de **300–400 MB**. Asigna **1 GB** de
límite en Dokploy y vas sobrado.

**Arranque en frío.** Si no incrustaste el `.parquet` en la imagen, el contenedor
descarga 31 MB la primera vez que alguien entra. Son unos segundos, una sola vez
por contenedor. Si prefieres que arranque instantáneo y sin depender de GitHub,
descomenta el `COPY secop2_limpio.parquet .` del Dockerfile: a cambio, los datos
quedan congelados hasta el siguiente despliegue.

**Enlaces compartibles.** La búsqueda y los filtros viajan en la dirección de la página, así que
cualquiera puede copiar el enlace de lo que encontró y mandarlo. Es la función más
importante de una herramienta de control ciudadano y conviene mencionarla al
difundirla.

**Dos límites que no se arreglan desde aquí:**

- **Google no va a indexar el contenido.** Streamlit dibuja la página por
  websocket, no hay HTML que un buscador pueda leer. Quien busque "contratos
  Cartagena" no va a llegar solo: la difusión tiene que ser por enlace directo,
  redes o prensa.
- **Cada clic es una petición al servidor.** Para tráfico normal no hay problema,
  pero una punta fuerte (una nota de prensa, un enlace que se viraliza) puede
  tumbar el contenedor justo el día que más importa. Si eso llega a ser una
  preocupación real, la salida es publicar un sitio estático precalculado —el
  análisis y los textos ya están separados de la interfaz precisamente para poder
  hacerlo sin rehacer el método.

## Notas sobre la imagen

- El contenedor corre como el usuario sin privilegios `observatorio` (uid 10001),
  no como root.
- El *healthcheck* consulta `/_stcore/health` con Python, sin instalar `curl`.
- El tema queda fijado en claro y el menú de despliegue oculto: es un sitio para
  visitantes, no un tablero de trabajo.
- Las versiones de las dependencias están fijadas en `requirements.txt`, así que
  dos builds del mismo commit producen el mismo entorno.

## Comprobaciones si algo falla

- **502 / Bad Gateway:** casi siempre es el *Container Port*. Debe ser **8501**.
- **La página carga pero queda “Please wait…” o no responde a clics:** es el
  websocket. Con Traefik suele funcionar solo; si usas otro proxy delante,
  asegúrate de reenviar los encabezados de *Upgrade/Connection* (websocket).
- **Aparece “No se pudieron cargar los datos de contratación”:** el contenedor no
  pudo bajar el parquet. El mensaje en pantalla dice si fue un timeout, un 404 o
  falta de red. Si tu servidor no tiene salida a internet, descomenta el `COPY` del
  `.parquet` en el Dockerfile y súbelo al repo.
- **Build lento o pesado:** normal la primera vez (instala pandas, sklearn,
  pyarrow). Los siguientes builds reutilizan caché si no cambia `requirements.txt`.

## Probar la imagen en local (opcional, si tienes Docker)

```bash
docker build -t observatorio .
docker run -p 8501:8501 observatorio
# abre http://localhost:8501
```
