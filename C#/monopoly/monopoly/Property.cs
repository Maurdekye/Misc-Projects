using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace monopoly
{
    abstract class Property : TradingTender
    {
        public Proprietor<Property> Owner;
        public Proprietor<Property> BoardLocation;
        public Game PlayingGame;
        public string Name { get; protected set; }

        public int Price { get; protected set; }
        public int MortgageValue { get; protected set; }
        public bool IsMortgaged { get; protected set; } = false;

        public bool CanUpgrade()
        {
            if (Owner.GetCapital() < GetUpgradePrice())
                return false;
            if (IsMortgaged)
                return true;
            return CanUpgrade_Internal();
        }
        protected abstract bool CanUpgrade_Internal();

        public int GetUpgradePrice()
        {
            if (IsMortgaged)
                return MortgageValue + (MortgageValue / 10);
            return GetUpgradePrice_Internal();
        }
        protected abstract int GetUpgradePrice_Internal();

        public void Upgrade()
        {
            if (!CanUpgrade())
            {
                throw new InvalidOperationException("Attempted to upgrade property when unable to");
            }
            Owner.ModifyCapital(-GetUpgradePrice());
            if (IsMortgaged)
            {
                IsMortgaged = false;
            }
            else
            {
                Upgrade_Internal();
            }
        }
        protected abstract void Upgrade_Internal();

        protected abstract bool IsUpgraded();

        public bool CanDowngrade()
        {
            return IsUpgraded() || !IsMortgaged;
        }

        public int GetDowngradeValue()
        {
            if (!IsUpgraded() && !IsMortgaged)
                return MortgageValue;
            return GetDowngradeValue_Internal();
        }
        protected abstract int GetDowngradeValue_Internal();

        public void Downgrade()
        {
            if (!CanDowngrade())
            {
                throw new InvalidOperationException("Attempted to downgrade property when unable to");
            }
            Owner.ModifyCapital(GetDowngradeValue());
            if (!IsUpgraded())
            {
                IsMortgaged = true;
            }
            else
            {
                Downgrade_Internal();
            }
        }
        protected abstract void Downgrade_Internal();

        public int GetRent(Player victim)
        {
            if (IsMortgaged)
            {
                return 0;
            }
            else
            {
                return GetRent_Internal(victim);
            }
        }

        public override string ToString()
        {
            return Name;
        }

        protected abstract int GetRent_Internal(Player victim);
    }

    enum Color
    {
        Brown, Blue, Pink, Orange, Red, Yellow, Green, Navy
    }

    class ColoredProperty : Property
    {
        public Color PropertyColor { get; private set; }
        private List<int> RentValues;
        public int HouseCost { get; private set; }

        public int Houses { get; private set; } = 0;

        public ColoredProperty(string Name, int Price, int MortgageValue, Color PropertyColor, List<int> RentValues, int HouseCost)
        {
            this.Name = Name;
            this.Price = Price;
            this.MortgageValue = MortgageValue;
            this.PropertyColor = PropertyColor;
            this.RentValues = RentValues;
            this.HouseCost = HouseCost;
        }

        protected override int GetRent_Internal(Player victim)
        {
            int baseRent = RentValues[Houses]*2;
            if (Houses > 0 || Util.FilterSubclass<Property, ColoredProperty>(victim.PlayingGame.PropertyList).Any(p => p.PropertyColor == PropertyColor && p.Owner != Owner))
            {
                baseRent /= 2;
            }
            return baseRent;
        }

        protected override bool CanUpgrade_Internal()
        {
            return Houses < 5;
        }

        protected override int GetUpgradePrice_Internal()
        {
            return HouseCost;
        }

        protected override void Upgrade_Internal()
        {
            Houses += 1;
        }

        protected override bool IsUpgraded()
        {
            return Houses > 0;
        }

        protected override int GetDowngradeValue_Internal()
        {
            return HouseCost / 2;
        }

        protected override void Downgrade_Internal()
        {
            Houses -= 1;
        }
    }

    class StationProperty : Property
    {
        public StationProperty(string Name, int Price, int MortgageValue)
        {
            this.Name = Name;
            this.Price = Price;
            this.MortgageValue = MortgageValue;
        }

        protected override bool CanUpgrade_Internal()
        {
            return false;
        }

        protected override void Downgrade_Internal()
        {
            throw new InvalidOperationException("Player should never be able to downgrade a station property beyond mortgaging it");
        }

        protected override int GetDowngradeValue_Internal()
        {
            return 0;
        }

        protected override int GetRent_Internal(Player victim)
        {
            return (2 ^ (victim.PlayingGame.PropertyList.Where(p => p is StationProperty && p.Owner == Owner).Count() - 1)) * 25;
        }

        protected override int GetUpgradePrice_Internal()
        {
            return 0;
        }

        protected override bool IsUpgraded()
        {
            return false;
        }

        protected override void Upgrade_Internal()
        {
            throw new InvalidOperationException("Player should never be able to upgrade a station property beyond mortgaging it");
        }
    }

    class UtilityProperty : Property
    {
        public UtilityProperty(string Name, int Price, int MortgageValue)
        {
            this.Name = Name;
            this.Price = Price;
            this.MortgageValue = MortgageValue;
        }

        protected override bool CanUpgrade_Internal()
        {
            return false;
        }

        protected override void Downgrade_Internal()
        {
            throw new InvalidOperationException("Player should never be able to downgrade a utility property beyond mortgaging it");
        }

        protected override int GetDowngradeValue_Internal()
        {
            return 0;
        }

        protected override int GetRent_Internal(Player victim)
        {
            int baseRent = Dice.Roll();
            if (victim.PlayingGame.PropertyList.Where(p => p is UtilityProperty && p.Owner == Owner).Count() == 2)
                return baseRent * 10;
            else
                return baseRent * 4;
        }

        protected override int GetUpgradePrice_Internal()
        {
            return 0;
        }

        protected override bool IsUpgraded()
        {
            return false;
        }

        protected override void Upgrade_Internal()
        {
            throw new InvalidOperationException("Player should never be able to upgrade a utility property beyond mortgaging it");
        }
    }
}
