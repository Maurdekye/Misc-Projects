
---  Prop Shooting Plugin
---  Created by Maurdekye

if SERVER then
	util.AddNetworkString( "choose_prop" )
	function try_spawn( ply, model ) 
		ply:SetNWString( "spawn_selection", model )
		net.Start( "choose_prop" )
			net.WriteString( model )
		net.Send( ply )
		--ply:SelectWeapon("prop_shooter")
		--return false
	end
	hook.Add( "PlayerSpawnRagdoll", "ragdoll_spawn_prevention", function ( ply, model )
		ply:SetNWBool( "is_ragdoll", true )
		return try_spawn(ply, model)
	end )
	hook.Add( "PlayerSpawnProp", "prop_spawn_prevention", function ( ply, model )
		ply:SetNWBool( "is_ragdoll", false )
		return try_spawn(ply, model)
	end )

	hook.Add( "PlayerSpawn", "give_prop_shooter", function( ply )
		ply:Give( "prop_shooter" )
	end )

elseif CLIENT then
	net.Receive( "choose_prop", function( len )
    	local modelname = string.match(net.ReadString(), ".-([^/]-)%.[^%.]-$")
		chat.AddText( Color( 200, 200, 200 ), "Selected ", Color( 255, 255, 255 ), modelname )
	end )
end