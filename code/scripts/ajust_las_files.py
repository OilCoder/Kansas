import os
import pandas as pd
import lasio
import zipfile
import json
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor

def read_csv_data(csv_path):
    return pd.read_csv(csv_path, on_bad_lines='skip')

def process_csv_data(csv_folder):
    files = os.listdir(csv_folder)
    wells_csv = next((f for f in files if f.startswith('Wells_')), None)
    tops_csv = next((f for f in files if f.startswith('Tops_')), None)
    logs_csv = next((f for f in files if f.startswith('Logs_')), None)
    LAS_csv = next((f for f in files if f.startswith('LAS_')), None)
    
    wells_data = read_csv_data(os.path.join(csv_folder, wells_csv)) if wells_csv else pd.DataFrame()
    tops_data = read_csv_data(os.path.join(csv_folder, tops_csv)) if tops_csv else pd.DataFrame()
    logs_data = read_csv_data(os.path.join(csv_folder, logs_csv)) if logs_csv else pd.DataFrame()
    LAS_data = read_csv_data(os.path.join(csv_folder, LAS_csv)) if LAS_csv else pd.DataFrame()

    return wells_data, tops_data, logs_data, LAS_data

def find_kid_for_las(las_file_name, LAS_data):
    kid_row = LAS_data[LAS_data['LASFILE'] == las_file_name]
    return kid_row['KID'].iloc[0] if not kid_row.empty else None

class CurvaSinDatosError(Exception):
    pass

def validar_curvas_y_lanzar_excepcion(las):
    for curve in las.curves:
        if curve.data.size == 0:  # Verifica si la curva está definida pero no tiene datos
            raise CurvaSinDatosError(f"La curva {curve.mnemonic} está definida pero no tiene datos.")
        
# Función para agregar errores al archivo de registro de manera acumulativa
# Modificación de la función log_error para recibir el nombre del campo directamente
def log_error(file_path, error_info, field_name, log_file_path='../reports/error_log.json'):
    # Extraer solo el número LAS del nombre del archivo para el registro
    las_number = os.path.basename(file_path).split('.')[0]

    try:
        # Intenta leer el archivo de registro existente
        with open(log_file_path, 'r') as file:
            existing_errors = json.load(file)
    except FileNotFoundError:
        # Si el archivo no existe, inicializa un diccionario vacío
        existing_errors = {}

    # Agrega el nuevo error al campo correspondiente
    if field_name not in existing_errors:
        existing_errors[field_name] = []
    
    error_entry = {
        "file": las_number,
        "error": error_info['error'],
        "message": error_info['message']
    }
    
    existing_errors[field_name].append(error_entry)
    
    # Guarda el diccionario actualizado de errores en el archivo
    with open(log_file_path, 'w') as file:
        json.dump(existing_errors, file, indent=4)

def update_las(las_file_path, well_data, log_data, top_data, LAS_data, destination_folder):
    try:
        las = lasio.read(las_file_path,  engine='normal')
        validar_curvas_y_lanzar_excepcion(las)
        kid = find_kid_for_las(os.path.basename(las_file_path), LAS_data)

        if kid:
            filtered_well_data = well_data[well_data['KID'] == kid]
            filtered_log_data = log_data[log_data['KID'] == kid]
            filtered_top_data = top_data[top_data['KID'] == kid]

            well_name = f"{filtered_well_data['LEASE_NAME'].iloc[0]} {filtered_well_data['WELL_NAME'].iloc[0]}"
            las.well['WELL'].value = well_name

            las.well['STRT'].value = filtered_log_data['TOP'].min() if not filtered_log_data.empty else None
            las.well['STOP'].value = filtered_log_data['BOTTOM'].max() if not filtered_log_data.empty else None

            if 'FLD' not in las.well:
                las.well['FLD'] = lasio.HeaderItem(mnemonic='FLD', unit='', value='', descr='Field')
            las.well['FLD'].value = filtered_well_data['FIELD_NAME'].iloc[0]

            if 'LOC' not in las.well:
                las.well['LOC'] = lasio.HeaderItem(mnemonic='LOC', unit='', value='', descr='Location')
            las.well['LOC'].value = filtered_log_data['LOCATION'].iloc[0]

            if 'CTRY' not in las.well:
                las.well['CTRY'] = lasio.HeaderItem(mnemonic='CTRY', unit='', value='', descr='Country')
            las.well['CTRY'].value = 'USA'

            if 'STAT' not in las.well:
                las.well['STAT'] = lasio.HeaderItem(mnemonic='STAT', unit='', value='', descr='State')
            las.well['STAT'].value = 'Kansas'

            if 'PROV' not in las.well:
                las.well['PROV'] = lasio.HeaderItem(mnemonic='PROV', unit='', value='', descr='Province')
            las.well['PROV'].value = filtered_well_data['TOWNSHIP'].iloc[0]

            if 'CNTY' not in las.well:
                las.well['CNTY'] = lasio.HeaderItem(mnemonic='CNTY', unit='', value='', descr='County')
            las.well['CNTY'].value = filtered_well_data['COUNTY'].iloc[0]

            if 'UWI' not in las.well:
                las.well['UWI'] = lasio.HeaderItem(mnemonic='UWI', unit='', value=kid, descr='Unique Well Identifier')

            if 'API' not in las.well:
                las.well['API'] = lasio.HeaderItem(mnemonic='API', unit='', value='', descr='API Number')
            las.well['API'].value = filtered_well_data['API'].iloc[0]

            if 'SRVC' not in las.well:
                las.well['SRVC'] = lasio.HeaderItem(mnemonic='SRVC', unit='', value='', descr='Logger Service Company')
            las.well['SRVC'].value = filtered_log_data['LOGGER'].iloc[0]

            if 'DATE' not in las.well:
                las.well['DATE'] = lasio.HeaderItem(mnemonic='DATE', unit='', value='', descr='Date logged')
            las.well['DATE'].value = filtered_log_data['LOG_DATE'].iloc[0]

            if 'XCOORD' not in las.well:
                las.well['XCOORD'] = lasio.HeaderItem(mnemonic='XCOORD', unit='NAD27', value='', descr='X Coordinate')
            las.well['XCOORD'].value = filtered_well_data['NAD27_LONGITUDE'].iloc[0]

            if 'YCOORD' not in las.well:
                las.well['YCOORD'] = lasio.HeaderItem(mnemonic='YCOORD', unit='NAD27', value='', descr='Y Coordinate')
            las.well['YCOORD'].value = filtered_well_data['NAD27_LATITUDE'].iloc[0]

            las = add_formation_info_to_las_other_section(las, filtered_top_data)

            # Dentro de tu función de actualización de archivos LAS
            base_filename = f"{filtered_well_data['LEASE_NAME'].iloc[0].replace(' ', '_').replace('/', '_')}_{filtered_well_data['WELL_NAME'].iloc[0].replace(' ', '_').replace('/', '_')}"
            extension = ".las"
            output_filename = f"{base_filename}{extension}"
            output_path = os.path.join(destination_folder, output_filename)

            # Inicializa un contador para duplicados
            duplicate_counter = 0

            # Verificar si el archivo ya existe para evitar sobrescritura
            while os.path.exists(output_path):
                duplicate_counter += 1
                output_filename = f"{base_filename}_duplicate_{duplicate_counter}{extension}"
                output_path = os.path.join(destination_folder, output_filename)

            las.write(output_path, version=2.0, fmt='%.4f', column_fmt={0: '%.2f'}, mnemonics_header=True)

        else:
            # No se encontró KID
            error_info = {
                'file': os.path.basename(las_file_path),
                'error': "KID not found",
                'message': "KID not found for file " + os.path.basename(las_file_path)
            }
            log_error(os.path.basename(las_file_path), error_info, filtered_well_data['FIELD_NAME'].iloc[0])  # Usa el nombre del campo determinado o "Unknown"
            
    except CurvaSinDatosError as e:
        error_info = {
            'file': las_file_path,
            'error': str(e),
            'message': "Curva definida sin datos encontrada."
        }
        log_error(las_file_path, error_info, filtered_well_data['FIELD_NAME'].iloc[0])  # Corregido

    except Exception as e:
        error_info = {
            'file': las_file_path,
            'error': str(e),
            'message': 'Error during the processing of the LAS file.'
        }
        log_error(las_file_path, error_info, filtered_well_data['FIELD_NAME'].iloc[0])  # Corregido

    except BaseException as e:
        error_info = {
            'file': las_file_path,
            'error': str(e),
            'message': 'Error during the processing of the LAS file.'
        }
        log_error(las_file_path, error_info, filtered_well_data['FIELD_NAME'].iloc[0])  # Corregido

    # Write the list of errors to a JSON file at the end of the function
    if error_log:
        log_file_path = os.path.join('../reports/','error_log.json')
        with open(log_file_path, 'w') as log_file:
            json.dump(error_log, log_file, indent=4)

def add_formation_info_to_las_other_section(las, top_data):

    # Preparar el encabezado CSV
    csv_header = "\tTOP, BASE, FORMATION\n"
    
    # Inicializar una lista para almacenar las líneas de datos de formación
    formation_lines = [csv_header]
    
    # Iterar a través del DataFrame de formaciones y formatear cada fila como CSV
    for index, row in top_data.iterrows():
        line = f"\t{row['TOP']}, {row['BASE']}, {row['FORMATION']}\n"
        formation_lines.append(line)
    
    # Unir todas las líneas en una sola cadena de texto
    formations_csv = "".join(formation_lines)
    
    # Añadir la cadena CSV formateada a la sección ~Other del archivo LAS
    if 'Other' in las.sections:
        las.sections['Other'] += formations_csv
    else:
        las.sections['Other'] = formations_csv
    
    return las

def process_zip_and_las_files(zip_file_path, wells_data, logs_data, tops_data, LAS_data, destination_folder):
    temp_dir = tempfile.mkdtemp()  # Crea un directorio temporal para extraer los archivos

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)  # Extrae todos los archivos del ZIP al directorio temporal
    
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.lower().endswith('.las'):
                las_file_path = os.path.join(root, file)
                if not wells_data.empty and not tops_data.empty and not logs_data.empty:
                    update_las(las_file_path, wells_data, logs_data, tops_data, LAS_data, destination_folder)
    
    shutil.rmtree(temp_dir)  # Limpia el directorio temporal

def process_las_files(source_folder, destination_folder, csv_folder):
    with ThreadPoolExecutor(max_workers=1) as executor:
        for field_folder_name in os.listdir(source_folder):
            field_source_folder = os.path.join(source_folder, field_folder_name)
            field_destination_folder = os.path.join(destination_folder, field_folder_name)
            field_csv_folder = os.path.join(csv_folder, field_folder_name)
            
            if os.path.isdir(field_source_folder) and os.path.isdir(field_csv_folder):
                wells_data, tops_data, logs_data, LAS_data = process_csv_data(field_csv_folder)
                
                if not os.path.isdir(field_destination_folder):
                    os.makedirs(field_destination_folder)
                
                for file_name in os.listdir(field_source_folder):
                    if file_name.lower().endswith('.zip'):
                        zip_file_path = os.path.join(field_source_folder, file_name)
                        # Enviar la tarea de procesar cada archivo ZIP al executor
                        executor.submit(process_zip_and_las_files, zip_file_path, wells_data, logs_data, tops_data, LAS_data, field_destination_folder)