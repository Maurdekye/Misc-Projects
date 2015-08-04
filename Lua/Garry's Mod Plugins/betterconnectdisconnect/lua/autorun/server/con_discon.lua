gameevent.Listen("player_disconnect")

util.AddNetworkString("playerConnect")
util.AddNetworkString("playerEnter")
util.AddNetworkString("playerDisconnect")
util.AddNetworkString("playerFailConnect")

ingame = {}

function connect(name, ip)
	sendNetMessage("playerConnect", name)
end
hook.Add("PlayerConnect", "PlayerConnect_con_discon", connect)

function firstspawn(ply)
	sendNetMessage("playerEnter", ply:GetName())
	ingame[ply:GetName()] = true
end
hook.Add("PlayerInitialSpawn", "PlayerInitialSpawn_con_discon", firstspawn)

function disconnect(data)
	if ingame[data.name] then
		sendNetMessage("playerDisconnect", data.name)
	else
		sendNetMessage("playerFailConnect", data.name)
	end
	ingame[data.name] = false
end
hook.Add("player_disconnect", "player_disconnect_con_discon", disconnect)

function sendNetMessage(msgType, name)
	net.Start(msgType)
		net.WriteString(name)
	net.Broadcast()
end

print(" -- Connect / Disconnect Messages Server Script Loaded --")
