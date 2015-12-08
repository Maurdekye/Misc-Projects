SWEP.Weight = 3
SWEP.AutoSwitchTo = true
SWEP.AutoSwitchFrom = false
SWEP.PrintName = "Super Crossbow"
SWEP.Slot = 3
SWEP.SlotPos = 5
SWEP.DrawAmmo = false
SWEP.DrawCrosshair = true

SWEP.Author = "Maurdekye"
SWEP.Purpose = "Super Fast Shooting Crossbow"
SWEP.Category = "Super Crossbow"
SWEP.Spawnable = true
SWEP.AdminOnly = true
SWEP.ViewModel = "models/weapons/v_crossbow.mdl"
SWEP.WorldModel = "models/weapons/w_crossbow.mdl"
SWEP.HoldType = "crossbow"

SWEP.Primary.ClipSize = -1
SWEP.Primary.DefaultClip = -1
SWEP.Primary.Damage = 100
SWEP.Primary.Cone = 0.5
SWEP.Primary.Automatic = true
SWEP.Primary.Ammo = "none"

SWEP.Secondary.ClipSize = -1
SWEP.Secondary.DefaultClip = -1
SWEP.Secondary.Automatic = false
SWEP.Secondary.Ammo = "none"

local shootSound = Sound("Weapon_Crossbow.Single")

function SWEP:PrimaryAttack()
	self.Weapon:SetNextPrimaryFire( CurTime() + 0.1 )	
	fire(self)
end

function fire(self)
	self:EmitSound(shootSound)

	if SERVER then
		local bolt = ents.Create("crossbow_bolt")
		if not IsValid(bolt) then return end

		local aimvec = self.Owner:GetAimVector() * 3500
		local cone = self.Primary.Cone
		randx = (math.random() - 0.5) * cone
		randy = (math.random() - 0.5) * cone
		aimvec:Rotate(Angle(0, randx, randy))

		bolt:SetPos( self.Owner:EyePos() + (self.Owner:GetAimVector() * 48) )
		bolt:SetAngles( aimvec:Angle() )
		bolt:SetOwner(self.Owner)
		bolt.damage = self.Primary.Damage
		bolt:Spawn()

		bolt:SetVelocity(aimvec)
		bolt.is_superxbow_bolt = true

		randx = (math.random() - 0.5) * 0.3
		randy = (math.random() + 0.3) * -0.5
		randz = (math.random() - 0.5) * 0.6
		self.Owner:ViewPunch(Angle(randy, randx, randz))
	end
end

function SWEP:TranslateFOV(old)
	self.currentfov = self.currentfov or self.Owner:GetFOV()
	if self.zoomed then
		self.currentfov = math.Approach(self.currentfov, 20, FrameTime()*50)
	else
		self.currentfov = math.Approach(self.currentfov, old, FrameTime()*20)
	end
	return self.currentfov
end

function SWEP:SecondaryAttack()
	if IsFirstTimePredicted() then
		self.zoomed = not self.zoomed
	end
	return false
end

function entTakeDamage(hit, info)
	inflict = info:GetInflictor()
	if inflict.is_superxbow_bolt then
		info:SetDamage(100)
		info:SetDamageForce(inflict:GetForward() * 40000)
	end
end
hook.Add("EntityTakeDamage", "EntityTakeDamage_supercrossbow", entTakeDamage)

print(" -- Super Crossbow Enabled --")
