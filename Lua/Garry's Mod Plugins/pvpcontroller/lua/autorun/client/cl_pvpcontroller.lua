b = Color(200, 200, 255)
bb = Color(150, 150, 255)
r = Color(255, 200, 200)
rr = Color(255, 150, 150)

bbb = Color(100, 100, 255)
rrr = Color(255, 100, 100)
g = Color(220, 220, 220)

function genericMessage(len)
	chat.AddText(Color(net.ReadInt(16), net.ReadInt(16), net.ReadInt(16)), net.ReadString())
end
net.Receive("genericMessage", genericMessage)

function plyNotFound(len)
	name = net.ReadString()
	chat.AddText(r, "Could not find a player by the name \"", g, name, r, "\".")
end
net.Receive("plyNotFound", plyNotFound)

function cantLeaveGod(len)
	left = net.ReadInt(32)
	nomencA = net.ReadString()
	nomencB = net.ReadString()
	chat.AddText(b, nomencA .. " must wait ", g, tostring(left), " seconds", b, " before leaving godmode.")
end
net.Receive("cantLeaveGod", cantLeaveGod)

function cantEnterGod(len)
	left = net.ReadInt(32)
	nomencA = net.ReadString()
	nomencB = net.ReadString()
	chat.AddText(r, nomencA .. " must wait ", g, tostring(left), " seconds", r, " before " .. nomencB .. " may leave the fray.")
end
net.Receive("cantEnterGod", cantEnterGod)

function enabledPVP(len)
	name = net.ReadString()
	chat.AddText(g, name, rr, " has ", rrr, "joined the fray, ", rr, "and is susceptible to damage!")
end
net.Receive("enabledPVP", enabledPVP)

function disabledPVP(len)
	name = net.ReadString()
	chat.AddText(g, name, bb, " has ", bbb, "left the fray.")
end
net.Receive("disabledPVP", disabledPVP)

function cantNoclip(len)
	chat.AddText(r, "You can't use noclip while in PvP mode; type ", g, "!god", r, ".")
end
net.Receive("cantNoclip", cantNoclip)