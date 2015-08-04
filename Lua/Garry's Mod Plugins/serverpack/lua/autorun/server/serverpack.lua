addonIds = {
	
	--[[
	["bike for scars 2.0"] = "106926982",
	["scars basic"] = "104487316",
	["scars extra"] = "104492363",
	["scars slim"] = "104483020",
	]]
	

	["fas 2.0 alpha sweps - misc"] = "201027186",
	["fas 2.0 alpha sweps - pistols"] = "181283903",
	["fas 2.0 alpha sweps - rifles"] = "181656972",
	["fas 2.0 alpha sweps - shotguns"] = "183140076",
	["fas 2.0 alpha sweps - smgs"] = "183139624",
	["fas 2.0 alpha sweps - u. rifles"] = "201027715",
	["fas 2.0 alpha sweps"] = "180507408",

	--[[
	["wac aircraft"] = "104990330",
	["wac community 1"] = "108907015",
	["wac community 2"] = "108909229",
	["wac community 3"] = "109977794",
	["wac community 4"] = "128559085",
	["wac community 5"] = "139224513",
	["wac community 6"] = "141486806",
	["wac community 7"] = "162016658",
	["wac halo"] = "178945066",
	["wac mh-x stealthhawk"] = "185480079",
	["wac roflcopter"] = "116014889",
	]]

	--["drivable yacht"] = "107590810",
	
	["nyan gun"] = "123277559",

	["playx v2.8.15"] = "106516163",
	
	["wiremod"] = "160250458"
}

c = 0
for k, v in pairs(addonIds) do
	PrintMessage(HUD_PRINTCONSOLE, "Mounting addon \"" .. k .. "\"")
	resource.AddWorkshop(v)
	c = c + 1
end
PrintMessage(HUD_PRINTCONSOLE, "Successfully mounted " .. c .. " serverside addons.")

concommand.Add("wire_activate_all_plugins", function(ply, cmd, args, argStr)
	if ply:IsAdmin() then
		for i, e in ipairs({"propcore", "constraintcore", "wiring", "effects", "remoteupload"}) do
			RunConsoleCommand("wire_expression2_extension_enable", e)
		end
		RunConsoleCommand("wire_expression2_reload")
	end
end)

concommand.Add("give_all_exclusive", function (ply, cmd, args, argstr)
	if ply:IsAdmin() and #args > 0 then
		for i,p in pairs(player.GetAll()) do
			p:StripWeapons()
			p:RemoveAllAmmo()
			p:Give(args[1])
			p:GiveAmmo(9999, p:GetWeapon(args[1]):GetPrimaryAmmoType(), true)
		end
	end
end)

concommand.Add("reset_loadout", function (ply, cmd, args, argstr)
	if ply:IsAdmin() then
		for i,p in pairs(player.GetAll()) do
			p:StripWeapons()
			p:RemoveAllAmmo()
			hook.Run("PlayerLoadout", p)
		end
	end
end)

hook.Add("PlayerLoadout", "PlayerLoadout_serverpack", function(ply)
	-- give default gmod accessories
	ply:Give("weapon_physgun")
	ply:Give("weapon_physcannon")
	ply:Give("gmod_tool")
	ply:Give("gmod_camera")

	-- give fa:s 2 weapons instead of hl2 weapons
	ply:Give("fas2_machete")

	ply:Give("fas2_glock20")
	ply:Give("fas2_ragingbull")

	ply:Give("fas2_mp5k")
	ply:Give("fas2_pp19")

	ply:Give("fas2_toz34")
	ply:Give("fas2_m82")
	ply:Give("fas2_m79")

	ply:Give("fas2_m67")

	return true
end)

local stroke = 5

timer.Simple((stroke * 60) - (os.time() % (stroke * 60)), function()
	timer.Create("generic_server_clock", stroke * 60, 0, function()
		print("\t\t\t\t" .. os.date("%X"))
	end)
end)

print("-- Generic Server Content Enabled --")