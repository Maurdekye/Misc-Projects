surface.CreateFont( "NameFont", {
	font = "Trebuchet18",
	size = 64,
	blursize = 1,
	antialias = true,
	})

hook.Add( "PostDrawOpaqueRenderables", "draw_names_admin", function()
	local ply = LocalPlayer()
	if ply:IsAdmin() then
		local plyPos = ply:EyePos()
		for _, p in pairs( player.GetAll() ) do
			if p ~= ply then
				local targetPos = p:EyePos()
                local scrndat = targetPos:ToScreen()
                if scrndat.visible then
                    cam.Start2D()
                        surface.SetFont( "NameFont" )
                        surface.SetTextColor( 200, 200, 200, 255 )
                        local w, h = surface.GetTextSize( p:Name() )
                        surface.SetTextPos( scrndat.x - (w/2), scrndat.y - h)
                        surface.DrawText( p:Name() )
                    cam.End2D()
                end
			end
		end
	end
end )