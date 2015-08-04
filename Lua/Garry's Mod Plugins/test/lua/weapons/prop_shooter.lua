
--- Prop Shooter SWEP
--- Created by Maurdekye

SWEP.Weight = 3
SWEP.AutoSwitchTo = false
SWEP.AutoSwitchFrom = true
SWEP.PrintName = "Prop Shooter"
SWEP.Slot = 5
SWEP.SlotPos = 1
SWEP.DrawAmmo = false
SWEP.DrawCrosshair = true

SWEP.Author = "Maurdekye"
SWEP.Purpose = "Use this to spawn props\nPress R to switch spawn mode"
SWEP.Category = "Other"
SWEP.Spawnable = true
SWEP.AdminOnly = false
SWEP.ViewModel = "models/weapons/v_rpg.mdl"
SWEP.WorldModel = "models/weapons/w_rocket_launcher.mdl"

SWEP.Primary.ClipSize = -1
SWEP.Primary.DefaultClip = -1
SWEP.Primary.Automatic = true
SWEP.Primary.Ammo = "none"

SWEP.Secondary.ClipSize = -1
SWEP.Secondary.DefaultClip = -1
SWEP.Secondary.Automatic = false
SWEP.Secondary.Ammo = "none"

local shootSound = Sound( "Weapon_Crossbow.Single" )
local clickSound = Sound( "Weapon_Pistol.Empty" )

local function r(m)
	return math.random() * 2 * m - m
end

function SWEP:PrimaryAttack()
	self.Weapon:SetNextPrimaryFire( CurTime() + 0.2 )
	local mult = fire_propshooter(self)
	self.Owner:SetVelocity(-self.Owner:GetAimVector() * mult)
end

function SWEP:SecondaryAttack()
	self.Weapon:SetNextSecondaryFire( CurTime() + 1 )
	local mult
	for i=1,7 do
		mult = fire_propshooter(self)
	end
	self.Owner:SetVelocity(-self.Owner:GetAimVector() * mult * 7)
end

function SWEP:Reload()
	if SERVER and IsFirstTimePredicted() then
		local timerid = "reload_stop_timer_" .. tostring( self.Owner:SteamID64() )
		if timer.Exists( timerid ) then
			timer.Adjust( timerid, 0.01, 1, function() end )
			return
		else
			timer.Create( timerid, 0.01, 1, function() end )
		end
		local response = {[true] = "enabled", [false] = "disabled"}
		local current = self.Owner:GetNWBool( "temp_props", false )
		self.Owner:SetNWBool( "temp_props", not current )
		self.Owner:PrintMessage(3, "Temporary props " .. response[not current])
	end
end

function fire_propshooter(self)
	if SERVER then
		local model = self.Owner:GetNWString( "spawn_selection" )
		if model ~= "" then

			local prop = nil
			local ragdoll = self.Owner:GetNWBool( "is_ragdoll" )
			if ragdoll then
				prop = ents.Create( "prop_ragdoll" )
			else
				prop = ents.Create( "physics_prop" )
			end
			if not IsValid( prop ) then
				self.Owner:EmitSound( clickSound )
				return
			end

			prop:SetModel( model )
			prop:SetPos( self.Owner:EyePos() + ( self.Owner:GetAimVector()  ) )
			prop:SetAngles( self.Owner:GetAimVector():Angle() )
			prop:Spawn()
			prop:Activate()
			prop:SetOwner(self.Owner)
			DoPropSpawnedEffect( prop )

			local phys = prop:GetPhysicsObject()
			if not IsValid( phys ) then
				self.Owner:EmitSound( clickSound )
				return
			else
				phys:Wake()
			end

			phys:SetVelocity( self.Owner:GetAimVector() * 4000 + self.Owner:GetVelocity() )
			phys:AddAngleVelocity( Vector( r(10), r(10), r(10) ) * 200 )


			if self.Owner:GetNWBool( "temp_props", false ) then
				timer.Simple( 60, function()
					if IsValid( prop ) then
						prop:Remove()
					end
				end )
			else
				if ragdoll then
					undo.Create( "Ragdoll" )
						undo.SetPlayer( self.Owner )
						undo.AddEntity( prop )
					undo.Finish( "Ragdoll (" .. model .. ")" )
					self.Owner:AddCleanup( "ragdolls", prop )
				else
					undo.Create( "Prop" )
						undo.SetPlayer( self.Owner )
						undo.AddEntity( prop )
					undo.Finish( "Prop (" .. model .. ")" )
					self.Owner:AddCleanup( "props", prop )
				end
			end

			randx = ( math.random() - 0.5 ) * 0.3
			randy = ( math.random() + 3 ) * -0.5
			randz = ( math.random() - 0.5 ) * 0.6
			self.Owner:ViewPunch( Angle( randy, randx, randz ) )

			self.Owner:EmitSound( shootSound )

			timer.Simple(0.4, function ()
				if IsValid( prop ) then
					prop:SetOwner()
				end
			end)

			return phys:GetMass()
		else
			self.Owner:EmitSound( clickSound )
		end
	end
	return 0
end