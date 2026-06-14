-- CoppeliaSim add-on: batch-export .ttm models to OBJ for RoboCasa import.
--
-- Usage (headless):
--   coppeliaSim.exe -h -b "<path>/coppelia_export_models.lua" -q -vinfos
--
-- Config: <repo>/exports/coppelia_edu/export_config.txt

function sysCall_info()
    return {
        autoStart = true,
    }
end

local function trim(s)
    return (s:gsub("^%s+", ""):gsub("%s+$", ""))
end

local function readConfig(configPath)
    local f, err = io.open(configPath, "r")
    if not f then
        error("Cannot open config: " .. tostring(configPath) .. " (" .. tostring(err) .. ")")
    end
    local cfg = {
        models_root = "",
        output_root = "",
        models = {},
    }
    for line in f:lines() do
        line = trim(line)
        if line ~= "" and not line:match("^#") then
            local key, val = line:match("^([^=]+)=(.*)$")
            if key then
                cfg[trim(key)] = trim(val)
            else
                table.insert(cfg.models, line)
            end
        end
    end
    f:close()
    if cfg.models_root == "" or cfg.output_root == "" then
        error("Config must set models_root= and output_root=")
    end
    return cfg
end

local function normalizePath(p)
    return p:gsub("\\", "/")
end

local function collectShapeHandles(modelHandle)
    local handles = sim.getObjectsInTree(modelHandle, sim.object_shape_type, 0)
    local exportHandles = {}
    for i = 1, #handles, 1 do
        local h = handles[i]
        local prop = sim.getObjectSpecialProperty(h)
        local renderable = sim.boolAnd32(prop, sim.objectspecialproperty_renderable)
        if renderable > 0 then
            local simple = sim.ungroupShape(h)
            if #simple > 0 then
                for j = 1, #simple, 1 do
                    exportHandles[#exportHandles + 1] = simple[j]
                end
            else
                exportHandles[#exportHandles + 1] = h
            end
        end
    end
    return exportHandles
end

local function exportObjAssimp(shapeHandles, outPath)
    if #shapeHandles == 0 then
        return false, "no renderable shapes"
    end
    if simAssimp == nil then
        return false, "simAssimp plugin not loaded"
    end
    local ok, err = pcall(function()
        simAssimp.exportShapes(shapeHandles, outPath, "obj", 1.0, simAssimp.upvect_z, 0)
    end)
    if not ok then
        return false, tostring(err)
    end
    return true, nil
end

local function exportObjLegacy(shapeHandles, outPath)
    if #shapeHandles == 0 then
        return false, "no renderable shapes"
    end
    local allVertices = {}
    local allIndices = {}
    for i = 1, #shapeHandles, 1 do
        local h = shapeHandles[i]
        local vertices, indices = sim.getShapeMesh(h)
        local m = sim.getObjectMatrix(h, -1)
        for vi = 1, #vertices // 3, 1 do
            local v = {
                vertices[3 * (vi - 1) + 1],
                vertices[3 * (vi - 1) + 2],
                vertices[3 * (vi - 1) + 3],
            }
            v = sim.multiplyVector(m, v)
            vertices[3 * (vi - 1) + 1] = v[1]
            vertices[3 * (vi - 1) + 2] = v[2]
            vertices[3 * (vi - 1) + 3] = v[3]
        end
        allVertices[#allVertices + 1] = vertices
        allIndices[#allIndices + 1] = indices
    end
    sim.exportMesh(0, outPath, 0, 1.0, allVertices, allIndices)
    return true, nil
end

local function exportUrdf(ttmPath, outDir)
    local modelHandle = sim.loadModel(ttmPath)
    if modelHandle == nil or modelHandle < 0 then
        return false, "sim.loadModel failed for " .. ttmPath
    end
    if simURDF == nil then
        sim.removeModel(modelHandle)
        return false, "simURDF plugin not loaded"
    end
    local outPath = outDir .. "/model.urdf"
    local ok, err = pcall(function()
        simURDF.export(modelHandle, outPath, 0)
    end)
    sim.removeModel(modelHandle)
    if not ok then
        return false, tostring(err)
    end
    return true, nil
end

local function exportModel(ttmPath, outDir)
    local modelHandle = sim.loadModel(ttmPath)
    if modelHandle == nil or modelHandle < 0 then
        return false, "sim.loadModel failed for " .. ttmPath
    end
    local shapeHandles = collectShapeHandles(modelHandle)
    local outPath = outDir .. "/model.obj"
    local ok, err = exportObjAssimp(shapeHandles, outPath)
    if not ok then
        ok, err = exportObjLegacy(shapeHandles, outPath)
    end
    sim.removeModel(modelHandle)
    return ok, err
end

local function resolveConfigPath()
    local appPath = sim.getStringParam(sim.stringparam_application_path)
    appPath = normalizePath(appPath)
    return appPath .. "/coppelia_export_config.txt"
end

local function runExport()
    local configPath = resolveConfigPath()
    print("[coppelia_export] config: " .. configPath)
    local cfg = readConfig(configPath)
    cfg.models_root = normalizePath(cfg.models_root)
    cfg.output_root = normalizePath(cfg.output_root)

    local logPath = cfg.output_root .. "/export_log.txt"
    local logFile = io.open(logPath, "w")

    local function log(msg)
        print("[coppelia_export] " .. msg)
        if logFile then
            logFile:write(msg .. "\n")
        end
    end

    if #cfg.models == 0 then
        log("No models listed in config; nothing to export.")
    end

    for _, rel in ipairs(cfg.models) do
        rel = normalizePath(rel)
        local exportMode = "mesh"
        if rel:match("^urdf:") then
            exportMode = "urdf"
            rel = rel:sub(6)
        end
        local ttmPath = cfg.models_root .. "/" .. rel
        local pathNoExt = rel:gsub("%.ttm$", "")
        local parent, leaf = pathNoExt:match("^(.+)/([^/]+)$")
        local stem = (leaf or pathNoExt):gsub(" ", "_")
        local outDir
        if parent then
            outDir = cfg.output_root .. "/" .. parent .. "/" .. stem
        else
            outDir = cfg.output_root .. "/" .. stem
        end
        log("Exporting [" .. exportMode .. "] " .. ttmPath .. " -> " .. outDir)
        local ok, err
        if exportMode == "urdf" then
            ok, err = exportUrdf(ttmPath, outDir)
        else
            ok, err = exportModel(ttmPath, outDir)
        end
        if ok then
            log("  OK")
        else
            log("  FAIL: " .. tostring(err))
        end
    end

    if logFile then
        logFile:close()
    end
end

function sysCall_init()
    local ok, err = pcall(runExport)
    if not ok then
        print("[coppelia_export] ERROR: " .. tostring(err))
    end
    sim.quitSimulator()
end
