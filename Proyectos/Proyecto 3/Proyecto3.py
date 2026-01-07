import pandas as pd
import os
from itertools import islice
from youtube_comment_downloader import YoutubeCommentDownloader

# --- 1. CONFIGURACIÓN ---
LIMIT_PER_VIDEO = 300  # Número máximo de comentarios a descargar por video


VIDEO_URLS = [
    "https://www.youtube.com/watch?v=luTkEGY5ixM",
    "https://www.youtube.com/watch?v=3xosRVxzjgA",
    "https://www.youtube.com/watch?v=ZMQALttUvyM",
    "https://www.youtube.com/watch?v=5qpPPM5NLNw",
    "https://www.youtube.com/watch?v=YXxBpGhlAeY",
    "https://www.youtube.com/watch?v=o2k7VjBUAtA",
    "https://www.youtube.com/watch?v=KlFCmHvWxlU",
    "https://www.youtube.com/watch?v=x7-sVvW2-5Y",
    "https://www.youtube.com/watch?v=QA9Wfyh0mFQ",
    "https://www.youtube.com/watch?v=ZKk5sJ6S5rY",
    "https://www.youtube.com/watch?v=0ggK1Qz7HnI",
    "https://www.youtube.com/watch?v=7Pq-S557XQU",
]

def descargar_comentarios():
    downloader = YoutubeCommentDownloader()
    dataset_data = [] 

    print("--- Iniciando descarga de comentarios ---")
    
    for url in VIDEO_URLS:
        print(f"Procesando: {url}")
        try:
            # sort_by=0 descarga los "Más recientes" primero
            comments = downloader.get_comments_from_url(url, sort_by=0)
            
            # Iteramos con el límite establecido (1500)
            for comment in islice(comments, LIMIT_PER_VIDEO):
                text = comment['text']
                # Limpieza: quitamos saltos de línea para que no rompan el CSV
                text_clean = text.replace('\n', ' ').strip()
                
                # Filtro: Ignorar comentarios muy cortos o vacíos
                if len(text_clean) > 30:
                    dataset_data.append({
                        "video_url": url,       # Útil para que la IA sepa el contexto
                        "comentario": text_clean,
                        "autor": comment.get('author', 'Anónimo')
                    })
                    
        except Exception as e:
            print(f"Error procesando video {url}: {e}")

    # --- GUARDADO ---
    
    if not dataset_data:
        print("No se extrajeron datos. Revisa tu conexión o las URLs.")
        return

    # 1. Crear DataFrame
    df = pd.DataFrame(dataset_data)
    
    # 2. Asegurar que existe la carpeta 'datos'
    if not os.path.exists('datos'):
        os.makedirs('datos')
        
    # 3. Guardar como CSV
    # Este formato es mejor para RAG/AnythingLLM que el .txt plano
    csv_filename = "datos/dataset_youtube_genz.csv"
    df.to_csv(csv_filename, index=False, encoding="utf-8")
            
    print(f"\n--- Éxito! ---")
    print(f"Se procesaron {len(VIDEO_URLS)} videos.")
    print(f"Total de comentarios guardados: {len(dataset_data)}")
    print(f"Archivo generado: {csv_filename}")

if __name__ == "__main__":
    descargar_comentarios()