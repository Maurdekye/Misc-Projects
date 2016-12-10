using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace monopoly
{
    abstract class BoardSpace
    {
        public Game PlayingGame;

        public BoardSpace() { }

        public abstract void Land(Player arrival);
    }

    class PropertySpace : BoardSpace, Proprietor<Property>
    {
        public PropertySpace(Property ThisProperty)
        {
            ThisProperty.Owner = this;
            ThisProperty.BoardLocation = this;
        }

        public IEnumerable<Property> GetProperties()
        {
            foreach (Property p in PlayingGame.PropertyList)
            {
                if (p.Owner == this)
                    yield return p;
            }
        }

        public int GetCapital()
        {
            throw new InvalidOperationException("Attempted to get capital from board space");
        }

        public void ModifyCapital(int amount)
        {
            throw new InvalidOperationException("Attempted to modify capital of board space");
        }

        public override void Land(Player arrival)
        {
            Property thisProperty = GetProperties().Single();
            arrival.Tell($"You landed on {thisProperty}.");
            if (thisProperty.Owner is Player)
            {
                Player propertyOwner = (Player)thisProperty.Owner;
                if (propertyOwner == arrival)
                {
                    arrival.Tell("You already own this property.");
                }
                else
                {
                    arrival.Tell($"This property is already owned by {propertyOwner}");
                    int rentCharge = thisProperty.GetRent(arrival);
                    if (thisProperty.IsMortgaged)
                        arrival.Tell("However, it is mortgaged, so you don't pay anything.");
                    else
                    {
                        arrival.Tell($"You pay {propertyOwner} a rent fee of ${rentCharge}");
                        arrival.Charge(rentCharge);
                    }
                }
            }
            else
            {
                arrival.Tell($"{thisProperty} is unowned.");
                if (arrival.AskToBuyProperty(thisProperty))
                {
                    arrival.Tell($"You have bought {thisProperty} for ${thisProperty.Price}");
                    arrival.Charge(thisProperty.Price);
                    thisProperty.Owner = arrival;
                }
                else
                {
                    arrival.Tell($"You decided not to buy {thisProperty}, so it will be auctioned off.");
                    arrival.PlayingGame.AuctionOffProperty(thisProperty);
                }
            }
        }
    }

    class JailSpace : BoardSpace
    {
        public override void Land(Player arrival)
        {
            arrival.Tell("You landed on jail, but you're just visiting.");
        }
    }

    class GotoJailSpace : BoardSpace
    {
        public override void Land(Player arrival)
        {
            arrival.Tell("You landed on Go Directly To Jail! You were sent to jail.");
            arrival.SendToJail();
        }
    }

    abstract class CardSpace : BoardSpace
    {
        public override void Land(Player arrival)
        {
            DrawCard(arrival);
        }

        public abstract void DrawCard(Player drawer);
    }

    class CommunityChestSpace : CardSpace
    {
        public override void DrawCard(Player drawer)
        {
            drawer.PlayingGame.CommunityChest.Draw(drawer);
        }
    }

    class ChanceSpace : CardSpace
    {
        public override void DrawCard(Player drawer)
        {
            drawer.PlayingGame.Chance.Draw(drawer);
        }
    }

    abstract class TaxSpace : BoardSpace { }

    class IncomeTaxSpace : TaxSpace
    {
        public override void Land(Player arrival)
        {
            int worth = arrival.CalculateTotalValueOfAssets();
            if (worth > 2000)
            {
                arrival.Tell($"You landed on income tax; 10% of your total worth is ${worth / 10}, so you opt to pay $200 instead");
            }
            else
            {
                arrival.Tell($"You landed on income tax; you pay 10% of your total worth at ${worth / 10}, instead of $200");
            }
            arrival.Charge(Math.Min(worth / 10, 200));
        }
    }

    class LuxuryTaxSpace : TaxSpace
    {
        public override void Land(Player arrival)
        {
            arrival.Tell("You landed on luxury tax; you pay $75.");
            arrival.Charge(75);
        }
    }

    class FreeParkingSpace : BoardSpace
    {
        public override void Land(Player arrival)
        {
            arrival.Tell("You landed on Free Parking.");
        }
    }

    class GoSpace : BoardSpace
    {
        public override void Land(Player arrival) { }
    }
}
