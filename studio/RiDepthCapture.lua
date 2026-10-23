local HttpService = game:GetService("HttpService")
local Selection = game:GetService("Selection")
local CoreGui = game:GetService("CoreGui")
local StarterGui = game:GetService("StarterGui")

local SERVER = "http://127.0.0.1:8765"
local GRID_WIDTH = 320
local GRID_HEIGHT = 180
local SAMPLES_PER_RUN = 400
local MAX_DISTANCE = 4000
local SETTLE_SECONDS = 0.25
local MIN_HEIGHT = 3
local MAX_HEIGHT = 40
local MIN_PITCH = -40
local MAX_PITCH = 12
local MIN_FOV = 55
local MAX_FOV = 85
local CLEARANCE = 1.5
local PAUSED_STATUS = 409
local PAUSE_RETRY_SECONDS = 1
local MAX_MISSES = 20

local toolbar = plugin:CreateToolbar("RiDepth")
local button = toolbar:CreateButton("Capture", "Capture depth training samples from this place", "rbxasset://textures/ui/GuiImagePlaceholder.png")
button.ClickableWhenViewportHidden = false

local running = false
local random = Random.new()

local function post(path, body)
	local encoded = HttpService:JSONEncode(body)
	local warned = false
	while true do
		local response = HttpService:RequestAsync({
			Url = SERVER .. path,
			Method = "POST",
			Headers = { ["Content-Type"] = "application/json" },
			Body = encoded,
		})
		if response.Success then
			return HttpService:JSONDecode(response.Body)
		end
		if response.StatusCode ~= PAUSED_STATUS or not running then
			error(path .. " returned " .. response.StatusCode .. ": " .. response.Body)
		end
		if not warned then
			warn("RiDepth paused: keep Roblox Studio as the active window and do not cover it")
			warned = true
		end
		task.wait(PAUSE_RETRY_SECONDS)
	end
end

local function showOverlay(color)
	local gui = Instance.new("ScreenGui")
	gui.Name = "RiDepthOverlay"
	gui.IgnoreGuiInset = true
	gui.DisplayOrder = 2147483647
	local frame = Instance.new("Frame")
	frame.Size = UDim2.fromScale(1, 1)
	frame.BackgroundColor3 = color
	frame.BorderSizePixel = 0
	frame.Parent = gui
	gui.Parent = CoreGui
	return gui
end

local function collectScene()
	local hidden = {}
	local minimum = Vector3.new(math.huge, math.huge, math.huge)
	local maximum = -minimum
	local found = false
	for _, item in workspace:GetDescendants() do
		if item:IsA("BasePart") then
			if item.Transparency >= 0.95 or not item.CanQuery then
				table.insert(hidden, item)
			elseif item.Size.Magnitude < 2048 then
				minimum = minimum:Min(item.Position)
				maximum = maximum:Max(item.Position)
				found = true
			end
		end
	end
	if not found then
		minimum = Vector3.new(-256, 0, -256)
		maximum = Vector3.new(256, 64, 256)
	end
	local params = RaycastParams.new()
	params.FilterType = Enum.RaycastFilterType.Exclude
	params.FilterDescendantsInstances = hidden
	params.IgnoreWater = false
	return params, minimum, maximum
end

local probe = Instance.new("Part")
probe.Name = "RiDepthProbe"
probe.Shape = Enum.PartType.Ball
probe.Size = Vector3.one * CLEARANCE * 2
probe.Anchored = true
probe.CanCollide = false
probe.CanQuery = false
probe.CanTouch = false
probe.Transparency = 1
probe.Archivable = false

local function isClear(origin, params)
	local overlap = OverlapParams.new()
	overlap.FilterType = Enum.RaycastFilterType.Exclude
	overlap.FilterDescendantsInstances = params.FilterDescendantsInstances
	probe.CFrame = CFrame.new(origin)
	probe.Parent = workspace.CurrentCamera
	local touching = workspace:GetPartsInPart(probe, overlap)
	probe.Parent = nil
	return #touching == 0
end

local function pickPose(params, minimum, maximum)
	for _ = 1, 50 do
		local x = random:NextNumber(minimum.X, maximum.X)
		local z = random:NextNumber(minimum.Z, maximum.Z)
		local top = maximum.Y + 500
		local ground = workspace:Raycast(Vector3.new(x, top, z), Vector3.new(0, -(top - minimum.Y + 1000), 0), params)
		if ground then
			local height = MIN_HEIGHT + (MAX_HEIGHT - MIN_HEIGHT) * random:NextNumber() ^ 2
			local origin = ground.Position + Vector3.new(0, height, 0)
			if isClear(origin, params) then
				local yaw = math.rad(random:NextNumber(0, 360))
				local pitch = math.rad(random:NextNumber(MIN_PITCH, MAX_PITCH))
				return CFrame.new(origin) * CFrame.Angles(0, yaw, 0) * CFrame.Angles(pitch, 0, 0)
			end
		end
	end
	return nil
end

local function traceDepth(camera, params)
	local viewport = camera.ViewportSize
	local look = camera.CFrame.LookVector
	local eye = camera.CFrame.Position
	local depth = table.create(GRID_WIDTH * GRID_HEIGHT, 0)
	local nx = table.create(GRID_WIDTH * GRID_HEIGHT, 0)
	local ny = table.create(GRID_WIDTH * GRID_HEIGHT, 0)
	local nz = table.create(GRID_WIDTH * GRID_HEIGHT, 0)
	for row = 0, GRID_HEIGHT - 1 do
		for column = 0, GRID_WIDTH - 1 do
			local ray = camera:ViewportPointToRay((column + 0.5) * viewport.X / GRID_WIDTH, (row + 0.5) * viewport.Y / GRID_HEIGHT)
			local result = workspace:Raycast(ray.Origin, ray.Direction * MAX_DISTANCE, params)
			local index = row * GRID_WIDTH + column + 1
			if result then
				depth[index] = math.floor((result.Position - eye):Dot(look) * 100 + 0.5)
				local normal = camera.CFrame:VectorToObjectSpace(result.Normal)
				nx[index] = math.floor(normal.X * 1000 + 0.5)
				ny[index] = math.floor(normal.Y * 1000 + 0.5)
				nz[index] = math.floor(normal.Z * 1000 + 0.5)
			end
		end
		if row % 6 == 0 then
			task.wait()
		end
	end
	return depth, nx, ny, nz
end

local function run()
	local camera = workspace.CurrentCamera
	local original = camera.CFrame
	local originalFov = camera.FieldOfView
	local originalGui = StarterGui.ShowDevelopmentGui
	StarterGui.ShowDevelopmentGui = false
	Selection:Set({})
	local params, minimum, maximum = collectScene()

	local overlay = showOverlay(Color3.fromRGB(255, 0, 255))
	task.wait(0.5)
	local ok, calibration = pcall(post, "/calibrate", { place = game.PlaceId, name = game.Name })
	overlay:Destroy()
	if not ok then
		warn("RiDepth calibration failed: " .. tostring(calibration))
		StarterGui.ShowDevelopmentGui = originalGui
		running = false
		button:SetActive(false)
		return
	end
	local session = calibration.session

	local captured = 0
	local misses = 0
	print("RiDepth is capturing, keep Studio in front and do not touch it until it reaches " .. SAMPLES_PER_RUN)
	while running and captured < SAMPLES_PER_RUN do
		local pose = pickPose(params, minimum, maximum)
		if not pose then
			misses += 1
			if misses >= MAX_MISSES then
				warn("RiDepth could not find open space to place the camera in this place, try a different place")
				break
			end
		else
			misses = 0
			camera.CFrame = pose
			camera.FieldOfView = random:NextNumber(MIN_FOV, MAX_FOV)
			task.wait(SETTLE_SECONDS)
			local id = session .. "_" .. captured
			local frameOk, frameError = pcall(post, "/frame", { id = id })
			if not frameOk then
				warn("RiDepth frame failed: " .. tostring(frameError))
				break
			end
			local depth, nx, ny, nz = traceDepth(camera, params)
			local depthOk, depthError = pcall(post, "/depth", {
				id = id,
				width = GRID_WIDTH,
				height = GRID_HEIGHT,
				fov = camera.FieldOfView,
				depth = depth,
				nx = nx,
				ny = ny,
				nz = nz,
			})
			if not depthOk then
				warn("RiDepth depth failed: " .. tostring(depthError))
				break
			end
			captured += 1
			if captured % 10 == 0 then
				print(string.format("RiDepth captured %d of %d", captured, SAMPLES_PER_RUN))
			end
		end
		task.wait()
	end

	camera.CFrame = original
	camera.FieldOfView = originalFov
	StarterGui.ShowDevelopmentGui = originalGui
	running = false
	button:SetActive(false)
	print("RiDepth capture finished with " .. captured .. " samples")
end

button.Click:Connect(function()
	running = not running
	button:SetActive(running)
	if running then
		task.spawn(run)
	end
end)
