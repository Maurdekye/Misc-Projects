--w = Color(255, 255, 255)

function blocktext(id, name, text, type)
	if type == "joinleave" then return true end
end
hook.Add( "ChatText", "ChatText_disable_join_discon", blocktext) 

function connectMessage(len)
	name = net.ReadString()
	chat.AddText(name, Color(150, 150, 255), " has connected to the server.")
end
net.Receive("playerConnect", connectMessage)

function enterMessage(len)
	name = net.ReadString()
	chat.AddText(name, Color(150, 255, 150), " has entered the game.")
end
net.Receive("playerEnter", enterMessage)

function disconnectMessage(len)
	name = net.ReadString()
	chat.AddText(name, Color(250, 100, 100), " has disconnected from the server.")
end
net.Receive("playerDisconnect", disconnectMessage)

function failConnectMessage(len)
	name = net.ReadString()
	chat.AddText(name, Color(200, 200, 0), " failed to join the game.")
end
net.Receive("playerFailConnect", failConnectMessage)

print(" -- Connect / Disconnect Messages Client Script Loaded --")