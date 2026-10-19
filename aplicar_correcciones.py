#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("ERROR: falta Pillow.")
    print("Instalalo con: py -m pip install Pillow")
    sys.exit(1)


def resolver_ruta(ruta_texto, carpeta_json, carpeta_actual):
    """Busca una ruta relativa junto al JSON o desde la carpeta actual."""
    ruta = Path(ruta_texto).expanduser()

    if ruta.is_absolute():
        return ruta

    opciones = [
        carpeta_json / ruta,
        carpeta_actual / ruta,
    ]

    for opcion in opciones:
        if opcion.is_file():
            return opcion

    return opciones[0]


def obtener_candidato(registro):
    """Devuelve el candidato elegido durante la revisión."""
    candidatos = registro.get("candidates", [])
    decision = registro.get("decision") or {}

    # Primero se utiliza la decisión explícita del HTML.
    indice = decision.get("candidate")

    if isinstance(indice, int) and 0 <= indice < len(candidatos):
        return candidatos[indice]

    # Compatibilidad con el campo selected_candidate.
    indice = registro.get("selected_candidate")

    if isinstance(indice, int) and 0 <= indice < len(candidatos):
        return candidatos[indice]

    # Último recurso: buscar por el nombre seleccionado.
    nombre = registro.get("selected_name")

    if nombre:
        for candidato in candidatos:
            if candidato.get("name") == nombre:
                return candidato

    return None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Genera imágenes PNG de las cartas seleccionadas "
            "con los nombres de las texturas de Duel Transer."
        )
    )

    parser.add_argument(
        "json",
        nargs="?",
        default="correcciones_duel_transer.json",
        help="JSON generado por el HTML de revisión."
    )

    parser.add_argument(
        "--salida",
        default="reemplazos_png",
        help="Carpeta donde guardar los PNG."
    )

    args = parser.parse_args()

    json_path = Path(args.json).expanduser().resolve()

    if not json_path.is_file():
        print(f"ERROR: no se encuentra el JSON:\n{json_path}")
        sys.exit(1)

    try:
        with json_path.open("r", encoding="utf-8-sig") as archivo:
            registros = json.load(archivo)
    except (json.JSONDecodeError, OSError) as error:
        print(f"ERROR al leer el JSON: {error}")
        sys.exit(1)

    if not isinstance(registros, list):
        print("ERROR: el JSON debe contener una lista de registros.")
        sys.exit(1)

    carpeta_json = json_path.parent
    carpeta_actual = Path.cwd()

    salida = Path(args.salida).expanduser()

    if not salida.is_absolute():
        salida = carpeta_actual / salida

    salida.mkdir(parents=True, exist_ok=True)

    creadas = 0
    omitidas = 0
    errores = 0

    print(f"JSON:   {json_path}")
    print(f"Salida: {salida.resolve()}")
    print("=" * 65)

    for numero, registro in enumerate(registros, start=1):
        if not isinstance(registro, dict):
            print(f"[OMITIDO] Registro {numero}: formato inválido.")
            omitidas += 1
            continue

        textura = registro.get("texture")

        if not textura:
            texture_path = registro.get("texture_path", "")
            textura = Path(texture_path).name if texture_path else None

        if not textura:
            print(f"[OMITIDO] Registro {numero}: no tiene nombre de textura.")
            omitidas += 1
            continue

        estado = str(registro.get("status", "")).lower()

        # No generar reemplazos para texturas marcadas como pendientes.
        if estado in {
            "unmatched",
            "pending",
            "unmatched_pending",
            "no_match"
        }:
            print(f"[PENDIENTE] {textura}")
            omitidas += 1
            continue

        # Si existe un estado, solo procesar las seleccionadas.
        if estado and estado != "selected":
            print(f"[OMITIDO] {textura}: estado '{estado}'.")
            omitidas += 1
            continue

        candidato = obtener_candidato(registro)

        if candidato is None:
            print(f"[OMITIDO] {textura}: no hay candidato seleccionado.")
            omitidas += 1
            continue

        ruta_texto = candidato.get("path")

        if not ruta_texto:
            filename = candidato.get("filename")

            if filename:
                ruta_texto = f"cartas_completas/{filename}"

        if not ruta_texto:
            print(f"[ERROR] {textura}: falta la ruta de la carta.")
            errores += 1
            continue

        origen = resolver_ruta(
            ruta_texto,
            carpeta_json,
            carpeta_actual
        )

        if not origen.is_file():
            print(f"[ERROR] No encuentro la imagen:")
            print(f"        {origen}")
            errores += 1
            continue

        # Usar solo el nombre del archivo para evitar rutas anidadas.
        destino = salida / Path(textura).name

        try:
            with Image.open(origen) as imagen:
                # Convertir formatos o modos no compatibles con PNG.
                if imagen.mode not in ("RGB", "RGBA"):
                    if "transparency" in imagen.info:
                        imagen = imagen.convert("RGBA")
                    else:
                        imagen = imagen.convert("RGB")

                imagen.save(destino, format="PNG")

            nombre = candidato.get(
                "name",
                registro.get("selected_name", origen.stem)
            )

            print(f"[OK] {textura}")
            print(f"     Carta: {nombre}")
            creadas += 1

        except Exception as error:
            print(f"[ERROR] No se pudo convertir {origen}")
            print(f"        {error}")
            errores += 1

    print("=" * 65)
    print(f"PNG creados:          {creadas}")
    print(f"Registros omitidos:   {omitidas}")
    print(f"Errores:              {errores}")
    print(f"Carpeta de resultados: {salida.resolve()}")

    if errores:
        sys.exit(2)


if __name__ == "__main__":
    main()