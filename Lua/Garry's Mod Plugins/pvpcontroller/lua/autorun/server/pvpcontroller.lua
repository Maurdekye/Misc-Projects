CreateConVar("pc_time_in_fray", 300, 0, "Time in seconds before a player can re-enter godmode")
CreateConVar("pc_recovery", 60, 0, "Time in seconds before a player can leave godmode again")
CreateConVar("pc_force_admin_pvp", 1, 0, "Whether or not admins are forced into pvp when they shoot other players")
CreateConVar("pc_admin_damage_god", 0, 0, "Whether or not admins can damage players in god mode")

util.AddNetworkString("plyNotFound")
util.AddNetworkString("genericMessage")
util.AddNetworkString("cantLeaveGod")
util.AddNetworkString("cantEnterGod")
util.AddNetworkString("enabledPVP")
util.AddNetworkString("disabledPVP")
util.AddNetworkString("cantNoclip")

b = Color(200, 200, 255)
bb = Color(150, 150, 255)
r = Color(255, 200, 200)
rr = Color(255, 150, 150)

function TIME_IN_FRAY() return GetConVarNumber("pc_time_in_fray") end
function RECOVERY() return GetConVarNumber("pc_recovery") end

metainfo = metainfo or {} 

-- hooks

function firstSpawn(ply)
	checkValues(ply)
end
hook.Add("PlayerInitialSpawn", "PlayerInitialSpawn_pvpcontroller", firstSpawn)

function preTakeDmg(ply, att)
	checkValues(ply)
	if att:IsPlayer() and att ~= ply then
		checkValues(att)
		if att:IsAdmin() and GetConVarNumber("pc_force_admin_pvp") == 0 then
			return not metainfo[ply:GetName()].god or GetConVarNumber("pc_admin_damage_god") == 1
		end
		ungod(att, false)
		if att.god then return false end
	end
	return not metainfo[ply:GetName()].god
end
hook.Add("PlayerShouldTakeDamage", "PlayerShouldTakeDamage_pvpcontroller", preTakeDmg)

function chat(ply, text, isTeamChat, isDead)
	checkValues(ply)
	sp = trim(text):split(" ")
	com = sp[1]:lower()
	stop = true
	isT = com == "!fray"
	target = ply
	nomencA = "You"
	nomencA2 = " are"
	nomencB = "you"
	force = false
	if ply:IsAdmin() and #sp > 1 and com:sub(1, 1) == "!" then
		conc = table.concat({unpack(sp, 2)}, " ")
		tarp = getPlayer(conc)
		if tarp == nil then
			notFoundPlayerMessage(ply, conc)
		else
			checkValues(tarp)
			tarp.time_out = 0
			target = tarp
			nomencA = target:GetName() 
			nomencA2 = " is"
			nomencB = "they"
			force = true
		end
	end
	if com == "!god" or (isT and !metainfo[ply:GetName()].god) then
		addgod(target, force)
	elseif com == "!ungod" or (isT and metainfo[ply:GetName()].god) then
		if metainfo[target:GetName()].god then
			ungod(target, force)
		else
			coloredMessage(ply, nomencA .. nomencA2 .. " already in PvP mode.", r)
		end
	elseif com == "!query" then
		if metainfo[target:GetName()].god then
			timeleft = math.floor((metainfo[target:GetName()].time_out + RECOVERY()) - CurTime())
			if timeleft > 0 then
				cantLeaveGod(ply, timeleft, nomencA, nomencB)
			else
				coloredMessage(ply, nomencA .. nomencA2 .. " currently in godmode, but " .. nomencB .. " may leave at any time.", b)
			end
		else
			timeleft = math.floor((metainfo[target:GetName()].time_out + TIME_IN_FRAY()) - CurTime())
			if timeleft > 0 then
				cantEnterGod(ply, timeleft, nomencA, nomencB)
			else
				coloredMessage(ply, nomencA .. nomencA2 .. " open to PvP, but " .. nomencB .. " may re-enable godmode at any time.", r)
			end
		end
	else
		stop = false
	end
	if stop then return false end
end
hook.Add("PlayerSay", "PlayerSay_pvpcontroller", chat)

local function noclip(ply, state)
	checkValues(ply)
	result = not (state and not metainfo[ply:GetName()].god)
	if not result then
		if ply:IsAdmin() then
			addgod(ply, false)
			return true
		else
			net.Start("cantNoclip")
			net.Send(ply)
		end
	end
	return result
end
hook.Add("PlayerNoClip", "PlayerNoClip_pvpcontrollery", noclip)

-- other

function ungod(ply, force)
	timeleft = math.floor((metainfo[ply:GetName()].time_out + RECOVERY()) - CurTime())
	if metainfo[ply:GetName()].god then
		if timeleft < 0 or ply:IsAdmin() or force then
			metainfo[ply:GetName()].god = false
			metainfo[ply:GetName()].time_out = CurTime()
			ply:SetHealth(100)
			ply:SetMoveType(MOVETYPE_WALK)
			coloredMessage(ply, "You have joined the fray!", rr)
			enabledPVP(ply)
		else
			cantLeaveGod(ply, timeleft)
		end
	else
		metainfo[ply:GetName()].time_out = CurTime()
	end
end

function addgod(ply, force)
	timeleft = math.floor((metainfo[ply:GetName()].time_out + TIME_IN_FRAY()) - CurTime())
	if not metainfo[ply:GetName()].god then
		if timeleft < 0 or ply:IsAdmin() or force then
			metainfo[ply:GetName()].god = true
			metainfo[ply:GetName()].time_out = CurTime()
			coloredMessage(ply, "You have left the fray.", bb)
			disabledPVP(ply)
		else
			cantEnterGod(ply, timeleft)
		end
	else
		coloredMessage(ply, "You're already in godmode.", b)
	end
end

function trim(s)
	return s:match("^%s*(.-)%s*$")
end

function checkValues(ply)
	metainfo[ply:GetName()] = metainfo[ply:GetName()] or {}
	if metainfo[ply:GetName()].god == nil then
		metainfo[ply:GetName()].god = true
		metainfo[ply:GetName()].time_out = -RECOVERY() - 1
	end
end

function getPlayer(name)
	name = string.lower(name);
	for _,v in pairs(player.GetAll()) do
		if string.find(string.lower(v:Name()),name,1,true) != nil then 
			return v;
		end
	end
end

function string:split(sep)
        local sep, fields = sep or ":", {}
        local pattern = string.format("([^%s]+)", sep)
        self:gsub(pattern, function(c) fields[#fields+1] = c end)
        return fields
end

-- network functions

function coloredMessage(ply, msg, color)
	net.Start("genericMessage")
		net.WriteInt(color.r, 16)
		net.WriteInt(color.g, 16)
		net.WriteInt(color.b, 16)
		net.WriteString(msg)
	net.Send(ply)
end

function notFoundPlayerMessage(ply, plyname)
	net.Start("plyNotFound")
		net.WriteString(plyname)
	net.Send(ply)
end

function cantLeaveGod(ply, timeleft, nomencA, nomencB)
	net.Start("cantLeaveGod")
		net.WriteInt(timeleft, 32)
		net.WriteString(nomencA or "You")
		net.WriteString(nomencB or "you")
	net.Send(ply)
end

function cantEnterGod(ply, timeleft, nomencA, nomencB)
	net.Start("cantEnterGod")
		net.WriteInt(timeleft, 32)
		net.WriteString(nomencA or "You")
		net.WriteString(nomencB or "you")
	net.Send(ply)
end

function enabledPVP(ply)
	net.Start("enabledPVP")
		net.WriteString(ply:GetName())
	net.Broadcast()
end

function disabledPVP(ply)
	net.Start("disabledPVP")
		net.WriteString(ply:GetName())
	net.Broadcast()
end

PrintMessage(2, " --- PvP Controller v1.1 Loaded --- ")