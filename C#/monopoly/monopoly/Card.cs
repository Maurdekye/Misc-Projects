using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace monopoly
{
    class Deck : Queue<Card>
    {
        public void Draw(Player drawer)
        {
            Card drawn = Dequeue();
            drawn.Act(drawer);
            if (drawn is GetOutOfJailCard)
                drawer.KeptCards.Add(drawn);
            else
                Enqueue(drawn);
        }
    }

    public enum DeckType
    {
        CommunityChest, Chance
    }

    delegate void CardAction(Player actor);

    class Card
    {
        public DeckType FromDeck;
        public CardAction Action;

        public Card(DeckType FromDeck, CardAction Action)
        {
            this.FromDeck = FromDeck;
            this.Action = Action;
        }

        public void Act(Player actor)
        {
            Action.Invoke(actor);
        }
    }

    class GetOutOfJailCard : Card, TradingTender
    {
        public GetOutOfJailCard(DeckType FromDeck, CardAction Action) : base(FromDeck, Action) { }
    }
}
