#!/bin/bash
# Script de limpieza ANTES de levantar Docker
# Ejecutar SIEMPRE antes de docker compose up

echo "=== Limpieza Pre-Inicio ==="
echo ""

# 1. Detener todos los contenedores
echo "1. Deteniendo contenedores..."
docker compose --profile all down 2>/dev/null
echo ""

# 2. Limpiar logs de Airflow COMPLETAMENTE
echo "2. Limpiando TODOS los logs de Airflow..."
if [ -d "./airflow/logs" ]; then
    before_size=$(du -sm ./airflow/logs 2>/dev/null | cut -f1)
    
    # Eliminar TODO excepto el directorio base
    find ./airflow/logs -mindepth 1 -delete 2>/dev/null
    
    freed_space=$before_size
    echo "   - Espacio liberado: ${freed_space} MB"
else
    echo "   - Directorio de logs no encontrado"
fi
echo ""

# 3. Limpiar archivos temporales
echo "3. Limpiando archivos temporales..."
if [ -d "./airflow/dags/data" ]; then
    temp_files=$(find ./airflow/dags/data -name "temp_*" 2>/dev/null)
    count=$(echo "$temp_files" | grep -c "temp_" 2>/dev/null || echo "0")
    
    if [ "$count" -gt 0 ]; then
        size=$(du -cm ./airflow/dags/data/temp_* 2>/dev/null | tail -1 | cut -f1)
        rm -f ./airflow/dags/data/temp_* 2>/dev/null
        echo "   - Archivos eliminados: $count"
        echo "   - Espacio liberado: ${size} MB"
    else
        echo "   - No hay archivos temporales"
    fi
fi
echo ""

# 4. Limpiar Docker agresivamente
echo "4. Limpiando Docker..."
echo "   - Contenedores detenidos..."
docker container prune -f 2>/dev/null
echo "   - Imagenes sin usar..."
docker image prune -f 2>/dev/null
echo "   - Volumenes sin usar..."
docker volume prune -f 2>/dev/null
echo "   - Build cache..."
docker builder prune -f 2>/dev/null
echo "   - Completado"
echo ""

# 5. Mostrar espacio disponible
echo "5. Espacio en disco:"
docker system df
echo ""

echo "=== Limpieza completada ==="
echo ""
echo "Ahora puedes ejecutar:"
echo "docker compose --profile all up -d"
echo ""
