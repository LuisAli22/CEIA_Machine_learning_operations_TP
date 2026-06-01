# Script de limpieza ANTES de levantar Docker
# Ejecutar SIEMPRE antes de docker compose up

Write-Host "=== Limpieza Pre-Inicio ===" -ForegroundColor Cyan
Write-Host ""

# 1. Detener todos los contenedores
Write-Host "1. Deteniendo contenedores..." -ForegroundColor Yellow
docker compose --profile all down 2>$null
Write-Host ""

# 2. Limpiar logs de Airflow COMPLETAMENTE
Write-Host "2. Limpiando TODOS los logs de Airflow..." -ForegroundColor Yellow
$airflowLogsPath = ".\airflow\logs"
if (Test-Path $airflowLogsPath) {
    $beforeSize = (Get-ChildItem -Path $airflowLogsPath -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    if ($null -eq $beforeSize) { $beforeSize = 0 }
    
    # Eliminar TODO excepto el directorio base
    Get-ChildItem -Path $airflowLogsPath -Recurse -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path $airflowLogsPath -Recurse -Directory -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    
    $freedSpaceMB = [math]::Round($beforeSize / 1MB, 2)
    Write-Host "   - Espacio liberado: $freedSpaceMB MB" -ForegroundColor Green
} else {
    Write-Host "   - Directorio de logs no encontrado" -ForegroundColor Red
}
Write-Host ""

# 3. Limpiar archivos temporales
Write-Host "3. Limpiando archivos temporales..." -ForegroundColor Yellow
$dataPath = ".\airflow\dags\data"
if (Test-Path $dataPath) {
    $tempFiles = Get-ChildItem -Path $dataPath -Filter "temp_*" -ErrorAction SilentlyContinue
    $freedSpace = 0
    
    foreach ($file in $tempFiles) {
        $freedSpace += $file.Length
        Remove-Item $file.FullName -Force -ErrorAction SilentlyContinue
    }
    
    $freedSpaceMB = [math]::Round($freedSpace / 1MB, 2)
    Write-Host "   - Archivos eliminados: $($tempFiles.Count)" -ForegroundColor Green
    Write-Host "   - Espacio liberado: $freedSpaceMB MB" -ForegroundColor Green
}
Write-Host ""

# 4. Limpiar Docker agresivamente
Write-Host "4. Limpiando Docker..." -ForegroundColor Yellow
Write-Host "   - Contenedores detenidos..." -ForegroundColor Gray
docker container prune -f 2>$null
Write-Host "   - Imagenes sin usar..." -ForegroundColor Gray
docker image prune -f 2>$null
Write-Host "   - Volumenes sin usar..." -ForegroundColor Gray
docker volume prune -f 2>$null
Write-Host "   - Build cache..." -ForegroundColor Gray
docker builder prune -f 2>$null
Write-Host "   - Completado" -ForegroundColor Green
Write-Host ""

# 5. Mostrar espacio disponible
Write-Host "5. Espacio en disco:" -ForegroundColor Yellow
docker system df
Write-Host ""

Write-Host "=== Limpieza completada ===" -ForegroundColor Green
Write-Host ""
Write-Host "Ahora puedes ejecutar:" -ForegroundColor Cyan
Write-Host "docker compose --profile all up -d" -ForegroundColor White
Write-Host ""
